import os
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == 'input' else SCRIPT_DIR

SRC = os.path.join(ROOT_DIR, 'input', 'FY27 WISE Mentorship.xlsx')
if not os.path.exists(SRC):
    SRC = os.path.join(ROOT_DIR, 'FY27 WISE Mentorship.xlsx')

OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f'Reading source file: {SRC}')

df = pd.read_excel(SRC, sheet_name='Cleaned Data')
df['row_id'] = df.index

# ---------- 1. Resolve duplicate/conflicting mentee submissions ----------
# Cross-checked each pair against their original raw application (Raw Application Data tab).
# In every case exactly one of the two cleaned rows matches the raw submission; the other
# is a cleanup artifact (swapped priority order or wrong Operating Unit). Keep the verified
# row, drop the incorrect one.
INCORRECT_DUP_ROW_IDS = [247, 276, 299, 386]  # the wrong row for each of the 4 people
dup_mentee_mask = df['row_id'].isin(INCORRECT_DUP_ROW_IDS)
manual_review_dupes = df[dup_mentee_mask].copy()
manual_review_dupes['Flag Reason'] = 'Incorrect duplicate row (did not match raw submission) - excluded, verified row kept'
df = df[~dup_mentee_mask].copy()

mentees = df[df['Mentor or Mentee'] == 'Mentee'].copy().reset_index(drop=True)
mentors = df[df['Mentor or Mentee'] == 'Mentor'].copy().reset_index(drop=True)

# ---------- 2. Topic category mapping (mentee label <-> mentor label) ----------
def map_mentor_area(val):
    if pd.isna(val):
        return None
    if 'Leadership Development' in val:
        return 'Leadership Development'
    if 'Career Growth' in val:
        return 'Career Growth & Transitions'
    if 'Work-Life Integration' in val:
        return 'Work-life Balance'
    if 'Strategic Planning & Goals' in val:
        return 'Strategic Development Planning'
    if 'Networking & Relationships' in val:
        return 'Networking & Relationships'
    if 'Personal Brand & Influence' in val:
        return 'Personal Brand & Visibility'
    return None

for c in ['Mentor Area 1', 'Mentor Area 2', 'Mentor Area 3']:
    mentors[c + ' (mapped)'] = mentors[c].apply(map_mentor_area)

mentors['mentor_topics'] = mentors[[c + ' (mapped)' for c in ['Mentor Area 1', 'Mentor Area 2', 'Mentor Area 3']]].apply(
    lambda r: set(x for x in r if x is not None), axis=1)
mentees['mentee_topics'] = mentees[['Mentee Priority 1', 'Mentee Priority 2', 'Mentee Priority 3']].apply(
    lambda r: set(x for x in r if pd.notna(x)), axis=1)

# ---------- 3. Mentor capacity ----------
def capacity(val):
    if val in ('Yes, I would love to have an additional mentee',
               'Yes, if needed to prevent mentees from being excluded'):
        return 2
    return 1

mentors['capacity'] = mentors['More than One'].apply(capacity)

# ---------- 4. Eligibility + scoring ----------
GRADE_NOGO = None
DOUBLE_UP_PENALTY = 30
UNMATCHED_SCORE = -100
INFEASIBLE_COST = 1_000_000

def score_pair(mentee, mentor):
    """Return (score, review_flag, reasons, breakdown) or None if hard no-go."""
    reasons = []
    g_mentee, g_mentor = mentee['Grade Level'], mentor['Grade Level']
    gap = g_mentor - g_mentee
    if gap <= 0 or gap >= 5:
        return None  # hard no-go: mentor not senior enough, or too big a gap
    if mentor['Operating Unit'] == mentee['Operating Unit'] or mentor['Org Area'] == mentee['Org Area']:
        return None  # hard no-go: same operating unit or same org area

    review = False

    # Grade gap scoring (softened at senior levels: gap of 1 counts as ideal when mentee is L8+)
    if gap in (2, 3):
        grade_score = 20
    elif gap == 1:
        grade_score = 20 if g_mentee >= 8 else 10
    else:  # gap == 4
        grade_score = 5
        review = True
        reasons.append('4-level grade gap - recommend review')

    # Location / time zone
    if mentor['Area'] == mentee['Area']:
        loc_score = 25
    elif mentor['Working Hours'] == mentee['Working Hours']:
        loc_score = 10
    else:
        loc_score = 0

    # Topic overlap
    overlap = len(mentee['mentee_topics'] & mentor['mentor_topics'])
    topic_score = overlap * 5

    score = grade_score + loc_score + topic_score
    breakdown = {'grade_gap': gap, 'grade_score': grade_score, 'loc_score': loc_score,
                 'topic_overlap': overlap, 'topic_score': topic_score}
    return score, review, reasons, breakdown

# ---------- 5. Build slot list (mentor slots, incl. 2nd slot for opted-in mentors) ----------
slots = []  # each: (mentor_row_id, slot_number)
for _, m in mentors.iterrows():
    slots.append((m['row_id'], 1))
    if m['capacity'] == 2:
        slots.append((m['row_id'], 2))

n_mentees = len(mentees)
n_slots = len(slots)
n_dummy = n_mentees  # guarantee feasibility - one 'unmatched' dummy per mentee
n_cols = n_slots + n_dummy

cost = np.full((n_mentees, n_cols), INFEASIBLE_COST, dtype=float)
meta = [[None] * n_cols for _ in range(n_mentees)]

mentor_by_id = mentors.set_index('row_id')

for i, mentee in mentees.iterrows():
    for j, (mentor_id, slot_num) in enumerate(slots):
        mentor = mentor_by_id.loc[mentor_id]
        result = score_pair(mentee, mentor)
        if result is None:
            continue
        base_score, review, reasons, breakdown = result
        eff_score = base_score - (DOUBLE_UP_PENALTY if slot_num == 2 else 0)
        cost[i, j] = -eff_score
        meta[i][j] = (mentor_id, slot_num, base_score, eff_score, review, reasons, breakdown)
    for k in range(n_dummy):
        j = n_slots + k
        cost[i, j] = -UNMATCHED_SCORE
        meta[i][j] = ('UNMATCHED', None, UNMATCHED_SCORE, UNMATCHED_SCORE, False, ['No eligible mentor found'], {})

row_ind, col_ind = linear_sum_assignment(cost)

# ---------- 6. Build results ----------
results = []
for i, j in zip(row_ind, col_ind):
    mentee = mentees.iloc[i]
    mentor_id, slot_num, base_score, eff_score, review, reasons, breakdown = meta[i][j]
    if mentor_id == 'UNMATCHED':
        results.append({
            'Mentee': mentee['Full name'], 'Mentee Email': mentee['Email'],
            'Mentor': None, 'Mentor Email': None,
            'Score': None, 'Needs Review': True,
            'Notes': 'UNMATCHED - no eligible mentor under current constraints'
        })
    else:
        mentor = mentor_by_id.loc[mentor_id]
        results.append({
            'Mentee': mentee['Full name'], 'Mentee Email': mentee['Email'],
            'Mentor': mentor['Full name'], 'Mentor Email': mentor['Email'],
            'Mentor 2nd Mentee?': slot_num == 2,
            'Score': base_score,
            'Grade Gap': breakdown.get('grade_gap'),
            'Grade Score': breakdown.get('grade_score'),
            'Location Score': breakdown.get('loc_score'),
            'Topic Overlap Count': breakdown.get('topic_overlap'),
            'Topic Score': breakdown.get('topic_score'),
            'Needs Review': review,
            'Notes': '; '.join(reasons) if reasons else ''
        })

results_df = pd.DataFrame(results)
matched_df = results_df[results_df['Mentor'].notna()].sort_values('Score', ascending=False)
unmatched_df = results_df[results_df['Mentor'].isna()]

print(f"Total mentees: {n_mentees}")
print(f"Total mentors: {len(mentors)} (slots incl. 2nd mentee: {n_slots})")
print(f"Matched: {len(matched_df)}")
print(f"Unmatched: {len(unmatched_df)}")
print(f"Flagged for review (4-level gap): {matched_df['Needs Review'].sum()}")
print(f"Double-up mentors used: {matched_df['Mentor 2nd Mentee?'].sum()}")
print()
print("Score distribution:")
print(matched_df['Score'].describe())

matched_df.to_csv(os.path.join(OUTPUT_DIR, 'matched.csv'), index=False)
unmatched_df.to_csv(os.path.join(OUTPUT_DIR, 'unmatched.csv'), index=False)
manual_review_dupes.to_csv(os.path.join(OUTPUT_DIR, 'manual_review_dupes.csv'), index=False)

# ---------- 7. Unmatched mentor analysis ----------
used_mentor_names = set(matched_df['Mentor'])
unmatched_mentors = mentors[~mentors['Full name'].isin(used_mentor_names)].copy()

# For each unmatched mentor, find their best-scoring mentee (even though that mentee is
# already taken) to gauge how close a hypothetical swap would be, and whether it's worth
# revisiting vs. simply notifying the mentor they weren't matched this cycle.
mentee_current_score = matched_df.set_index('Mentee')['Score'].to_dict()
mentee_current_mentor = matched_df.set_index('Mentee')['Mentor'].to_dict()

best_hypothetical = []
for _, mentor in unmatched_mentors.iterrows():
    best = None  # (score, mentee_name)
    for _, mentee in mentees.iterrows():
        result = score_pair(mentee, mentor)
        if result is None:
            continue
        base_score = result[0]
        if best is None or base_score > best[0]:
            best = (base_score, mentee['Full name'])
    if best is None:
        best_hypothetical.append({
            'Mentor': mentor['Full name'], 'Mentor Email': mentor['Email'],
            'Grade Level': mentor['Grade Level'],
            'Historically accepted (L8+)': mentor['Grade Level'] >= 8,
            'Best possible mentee': None, 'Best possible score': None,
            "That mentee's current match": None, "That mentee's current score": None,
            'Score delta (hypothetical - current)': None,
            'Notes': 'No eligible mentee exists for this mentor under current rules'
        })
    else:
        best_score, best_mentee = best
        current_mentor = mentee_current_mentor.get(best_mentee)
        current_score = mentee_current_score.get(best_mentee)
        best_hypothetical.append({
            'Mentor': mentor['Full name'], 'Mentor Email': mentor['Email'],
            'Grade Level': mentor['Grade Level'],
            'Historically accepted (L8+)': mentor['Grade Level'] >= 8,
            'Best possible mentee': best_mentee, 'Best possible score': best_score,
            "That mentee's current match": current_mentor, "That mentee's current score": current_score,
            'Score delta (hypothetical - current)': (best_score - current_score) if current_score is not None else None,
            'Notes': ''
        })

unmatched_mentor_df = pd.DataFrame(best_hypothetical).sort_values(
    ['Historically accepted (L8+)', 'Score delta (hypothetical - current)'], ascending=[False, False]
)
unmatched_mentor_df.to_csv(os.path.join(OUTPUT_DIR, 'unmatched_mentors.csv'), index=False)

print()
print(f"Total mentors used: {len(used_mentor_names)} of {len(mentors)}")
print(f"Unmatched mentors: {len(unmatched_mentors)}")
print(f"  - of which L8+ (historically accepted grade): {(unmatched_mentors['Grade Level'] >= 8).sum()}")
print(f"  - of which L7 and below (historically not accepted): {(unmatched_mentors['Grade Level'] < 8).sum()}")
