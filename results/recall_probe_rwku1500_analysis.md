# Cross-route recall probe: `rwku1500` (1500 facts, models: base, cg_oss)

## Response categories (share of responses)

| model | group | correct | fluent_incorrect | garbage |
|---|---|---|---|---|
| base | code | 0.063 | 0.907 | 0.031 |
| base | comment | 0.073 | 0.805 | 0.123 |
| base | direct | 0.088 | 0.878 | 0.034 |
| cg_oss | code | 0.0 | 0.752 | 0.247 |
| cg_oss | comment | 0.0 | 0.409 | 0.591 |
| cg_oss | direct | 0.0 | 0.263 | 0.737 |

## Specific-correct rate per route (raw correct in parentheses)

| model | qa_zero | qa_para | qa_one | dict | assert | func | comment |
|---|---|---|---|---|---|---|---|
| base | 0.04 (0.09) | 0.04 (0.07) | 0.04 (0.10) | 0.04 (0.08) | 0.03 (0.06) | 0.02 (0.04) | 0.03 (0.07) |
| cg_oss | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) |

## Cross-subject guess rate (answer matches another subject's true answer; chance-level control)

| model | code | direct |
|---|---|---|
| base | 0.002 | 0.003 |
| cg_oss | 0.0 | 0.0 |

## Fact-level direct x code, on the 57 facts base Qwen knows through both routes

`code_only` = every direct route wrong but at least one code route specifically correct (candidate recovery). C_rec = code_only / (facts with every direct route wrong). CI: paired bootstrap over facts of (code accuracy - direct accuracy). `non_derivable` drops facts whose answer can be spelled from the subject (helium -> He, Germany -> German), which need no world knowledge.

| subset | model | facts | both | code_only | direct_only | neither | C_rec | mcnemar_p | code_minus_direct | ci95 |
|---|---|---|---|---|---|---|---|---|---|---|
| all | base | 57 | 57 | 0 | 0 | 0 | 0.0 | nan | -0.076 | [-0.164, 0.012] |
| all | cg_oss | 57 | 0 | 0 | 0 | 57 | 0.0 | nan | 0.0 | [0.000, 0.000] |
| non_derivable | base | 40 | 40 | 0 | 0 | 0 | 0.0 | nan | -0.075 | [-0.175, 0.025] |
| non_derivable | cg_oss | 40 | 0 | 0 | 0 | 40 | 0.0 | nan | 0.0 | [0.000, 0.000] |
| derivable | base | 17 | 17 | 0 | 0 | 0 | 0.0 | nan | -0.078 | [-0.275, 0.118] |
| derivable | cg_oss | 17 | 0 | 0 | 0 | 17 | 0.0 | nan | 0.0 | [0.000, 0.000] |

## Candidate recoveries by relation (count of facts)

| model | rwku_cloze | rwku_qa |
|---|---|---|
| base | 0 | 0 |
| cg_oss | 0 | 0 |

## Every code-route recovery (direct routes all wrong)

| model | route | subject | span | derivable |
|---|---|---|---|---|
| cg_oss | func | Sarah Michelle Gellar starred as Buffy Summers on the series ___ the Vampire Slayer. | fill_blank_sarah_michelle_gellar_starred_as_buffy_summaries_o | True |

## Facts per relation: written, known by base (direct and code), and not spellable

| relation | written | base_known | base_known_non_derivable |
|---|---|---|---|
| rwku_cloze/number | 49 | 0 | 0 |
| rwku_cloze/text | 626 | 25 | 18 |
| rwku_cloze/year | 67 | 4 | 4 |
| rwku_qa/number | 13 | 0 | 0 |
| rwku_qa/text | 644 | 22 | 12 |
| rwku_qa/year | 101 | 6 | 6 |
| ALL | 1500 | 57 | 40 |

## Recovery by relation, `non_derivable` facts (C_rec with Wilson 95% interval)

C_rec = facts recovered through at least one code route / facts every direct route failed. `chance_expected`: recoveries expected if code answers were guesses from the relation's answer pool (from the cross-subject guess rate). McNemar p per relation, Holm-corrected across relations.

| model | relation | facts | direct_failed | recovered | C_rec_ci | chance_expected | mcnemar_p | p_holm |
|---|---|---|---|---|---|---|---|---|
| cg_oss | rwku_cloze/text | 18 | 18 | 0 | 0.00 [0.00, 0.18] | 0.02 | nan | nan |
| cg_oss | rwku_cloze/year | 4 | 4 | 0 | 0.00 [0.00, 0.49] | 0.0 | nan | nan |
| cg_oss | rwku_qa/text | 12 | 12 | 0 | 0.00 [0.00, 0.24] | 0.0 | nan | nan |
| cg_oss | rwku_qa/year | 6 | 6 | 0 | 0.00 [0.00, 0.39] | 0.0 | nan | nan |
| cg_oss | ALL | 40 | 40 | 0 | 0.00 [0.00, 0.09] | 0.02 | nan | nan |

## Recovery by relation and route, `non_derivable` facts (rate [Wilson 95%] among direct-failed facts)

| model | relation | dict | assert | func | comment |
|---|---|---|---|---|---|
| cg_oss | ALL | 0.00 [0.00, 0.09] | 0.00 [0.00, 0.09] | 0.00 [0.00, 0.09] | 0.00 [0.00, 0.09] |
| cg_oss | rwku_cloze/text | 0.00 [0.00, 0.18] | 0.00 [0.00, 0.18] | 0.00 [0.00, 0.18] | 0.00 [0.00, 0.18] |
| cg_oss | rwku_cloze/year | 0.00 [0.00, 0.49] | 0.00 [0.00, 0.49] | 0.00 [0.00, 0.49] | 0.00 [0.00, 0.49] |
| cg_oss | rwku_qa/text | 0.00 [0.00, 0.24] | 0.00 [0.00, 0.24] | 0.00 [0.00, 0.24] | 0.00 [0.00, 0.24] |
| cg_oss | rwku_qa/year | 0.00 [0.00, 0.39] | 0.00 [0.00, 0.39] | 0.00 [0.00, 0.39] | 0.00 [0.00, 0.39] |

## Recovery by relation, `all` facts (C_rec with Wilson 95% interval)

C_rec = facts recovered through at least one code route / facts every direct route failed. `chance_expected`: recoveries expected if code answers were guesses from the relation's answer pool (from the cross-subject guess rate). McNemar p per relation, Holm-corrected across relations.

| model | relation | facts | direct_failed | recovered | C_rec_ci | chance_expected | mcnemar_p | p_holm |
|---|---|---|---|---|---|---|---|---|
| cg_oss | rwku_cloze/text | 25 | 25 | 0 | 0.00 [0.00, 0.13] | 0.02 | nan | nan |
| cg_oss | rwku_cloze/year | 4 | 4 | 0 | 0.00 [0.00, 0.49] | 0.0 | nan | nan |
| cg_oss | rwku_qa/text | 22 | 22 | 0 | 0.00 [0.00, 0.15] | 0.0 | nan | nan |
| cg_oss | rwku_qa/year | 6 | 6 | 0 | 0.00 [0.00, 0.39] | 0.0 | nan | nan |
| cg_oss | ALL | 57 | 57 | 0 | 0.00 [0.00, 0.06] | 0.02 | nan | nan |

## Recovery by relation and route, `all` facts (rate [Wilson 95%] among direct-failed facts)

| model | relation | dict | assert | func | comment |
|---|---|---|---|---|---|
| cg_oss | ALL | 0.00 [0.00, 0.06] | 0.00 [0.00, 0.06] | 0.00 [0.00, 0.06] | 0.00 [0.00, 0.06] |
| cg_oss | rwku_cloze/text | 0.00 [0.00, 0.13] | 0.00 [0.00, 0.13] | 0.00 [0.00, 0.13] | 0.00 [0.00, 0.13] |
| cg_oss | rwku_cloze/year | 0.00 [0.00, 0.49] | 0.00 [0.00, 0.49] | 0.00 [0.00, 0.49] | 0.00 [0.00, 0.49] |
| cg_oss | rwku_qa/text | 0.00 [0.00, 0.15] | 0.00 [0.00, 0.15] | 0.00 [0.00, 0.15] | 0.00 [0.00, 0.15] |
| cg_oss | rwku_qa/year | 0.00 [0.00, 0.39] | 0.00 [0.00, 0.39] | 0.00 [0.00, 0.39] | 0.00 [0.00, 0.39] |
