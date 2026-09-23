# Aspect codebook v1 (draft)

Status: draft written by the project assistant. The owner finalizes it.

Each review gets two binary flags per aspect: `pos_<aspect>` and `neg_<aspect>`.

- Both 0: the aspect is not evaluated in the review.
- `pos` = 1: the reviewer expresses a favourable judgement about the aspect.
- `neg` = 1: the reviewer expresses an unfavourable judgement about the aspect.
- Both 1: mixed, e.g. "coffee was great but the cold brew was watery".

Conventions follow the spirit of the SemEval-2014 Task 4 restaurant annotation
scheme (aspect categories with positive, negative, neutral, and conflict
polarity; implicit aspect mentions are annotated). "Conflict" maps to both
flags = 1. "Neutral" mentions (the aspect is named with no judgement, e.g.
"we ordered pasta") get both flags = 0. Source: guidelines as summarised by
search results; the guideline PDF itself was not reachable from the build
environment. TODO(citation): confirm against the PDF.

## General rules

1. Judge the reviewer's evaluation, not the mere presence of a word.
2. Implicit mentions count: "we were seated in 2 minutes" is `pos_wait_time`.
3. Recommendations and intentions ("will come again") are not aspect judgements.
4. Overall statements ("great place", "worst experience") are not assigned to any aspect.
5. Comparisons and expectations count in the direction stated: "better than Starbucks" about coffee is `pos_coffee`.
6. Sarcasm is coded by its intended meaning when it is clear; otherwise leave the aspect at 0.
7. Hearsay ("my friend said the food is bad") is not coded.

## Aspects

| Aspect | Covers | Examples (pos / neg) |
|---|---|---|
| `coffee` | Coffee and coffee-like drinks: filter coffee, espresso drinks, cold brew, tea when served as the cafe's hot beverage | "perfect cappuccino" / "coffee was bitter" |
| `food` | All food and non-coffee drinks: taste, quality, portion, freshness, temperature, menu variety | "pasta was delicious" / "sandwich was stale" |
| `service_staff` | Staff behaviour and service quality: friendliness, attentiveness, order accuracy, manager response | "staff were courteous" / "rude waiter", "wrong order" |
| `wait_time` | Speed and waiting: time to be seated, served, or delivered | "served quickly" / "waited 45 minutes" |
| `value` | Price and value for money | "reasonably priced" / "overpriced", "not worth it" |
| `ambience` | Decor, music, lighting, vibe, view, overall atmosphere | "cozy ambience" / "dull interiors" |
| `seating_space` | Seating comfort, space, table availability | "comfortable seating" / "cramped", "no place to sit" |
| `noise_crowding` | Noise and crowd levels | "quiet place" / "too loud", "overcrowded" |
| `wifi_work` | Wi-Fi, charging points, suitability to work or study | "good wifi" / "no charging points" |
| `cleanliness` | Hygiene of premises, tables, cutlery, washrooms, food safety (hair in food) | "clean and hygienic" / "dirty tables", "found a fly" |
| `location_access` | Location, parking, how easy the place is to find or reach | "easy to find" / "no parking" |

## Edge cases

- Delivery reviews: packaging and delivery time go to `wait_time` (time) or `service_staff` (packaging, missing items).
- Quantity: "portion was small" is `neg_food`. "Small portion for the price" is also `neg_value`.
- Menu variety ("limited options") is `neg_food`.
- Beverages other than coffee (shakes, mocktails, juices) are `food`.
- "Service" alone with a judgement is `service_staff`; "slow service" is `neg_wait_time` and, if the complaint is about staff behaviour too, also `neg_service_staff`.
