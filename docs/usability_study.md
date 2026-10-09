# Usability study protocol

**Question:** can non-technical users correctly read an issue's mood and trend from the
Overview page, within about 30 seconds per issue?

## Participants

* At least 10 adults with no statistics or programming background. Aim for a
  mix of age, gender, education, and urban and rural backgrounds.
* Obtain informed consent (template below). Record no names, only participant codes
  (P01, P02, …).
* Participants must not have seen the dashboard before.

## Setup

* Use **demo mode** (`VOTERPULSE_DEMO_MODE=1`) so every participant sees the same,
  known data. Regenerate it with the same seed (`generate_demo_data.py --seed 7`) on the
  day of the study, then note the correct answers for each card *before* the sessions.
* Laptop with a standard browser at 100% zoom, or a phone, recorded as a session variable.
* The facilitator reads tasks aloud from the script and does not help.

## Tasks (repeat for 3 issues per participant, rotating which issues)

1. "Look at the card for **{issue}**. In your own words, is news coverage of this issue
   mostly positive, mostly negative, mixed, or is there not enough information?"
2. "Is the coverage getting more positive, more negative, or staying about the same?"
3. "Roughly how many news headlines is this based on?"
4. (Once per participant) "Does this page tell you what voters think?"
   The correct answer is no.

Start the timer when the facilitator finishes reading task 1 and stop it when the
participant has answered tasks 1 and 2.

## Recording sheet

| Participant | Device | Issue | Mood answer | Correct? | Trend answer | Correct? | Count answer | Within ±20%? | Seconds (tasks 1–2) | Disclaimer understood? | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P01 | | | | | | | | | | | |

Scoring: a mood answer is correct when it matches the card's label, accepting
synonyms ("bad" = negative). Record the participant's exact words in Notes.

## Post-session questions (5-point scale, 1 = strongly disagree)

1. I understood what the coloured labels meant.
2. I would trust this to tell me how an issue is being covered in the news.
3. It was clear that this does not measure what voters think.
4. I could find the actual headlines behind a statement if I wanted to.
5. Open question: "What was confusing?"

## Analysis

* Mood accuracy: % correct, with 95% Wilson interval. Same for trend.
* Median and 90th-percentile time per issue; % of answers within 30 seconds.
* % of participants who correctly answer the disclaimer question.
* Group the open-ended comments into themes, and list each concrete change made
  in response.

## Report template

```
Participants: n=__ (devices: __ laptop / __ phone). Dates: __.
Mood interpretation accuracy: __% (95% CI __–__%), trend: __% (__–__%).
Median time per issue: __ s; __% within 30 s.
Disclaimer understood: __/__.
Main confusions: …
Changes made: …
```

## Consent template (summary)

"You are invited to try a prototype website that summarises the tone of news headlines.
We are testing the website, not you. The session takes about 15 minutes. We record your
answers and the time taken, but not your name. You may stop at any time without
giving a reason."
