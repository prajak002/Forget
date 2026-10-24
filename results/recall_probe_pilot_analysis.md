# Cross-route recall probe: `pilot` (20 facts, models: base, cg_oss, max_entropy, npo)

## Response categories (share of responses)

| model | group | correct | fluent_incorrect | garbage |
|---|---|---|---|---|
| base | code | 0.917 | 0.083 | 0.0 |
| base | comment | 0.95 | 0.0 | 0.05 |
| base | direct | 0.95 | 0.05 | 0.0 |
| cg_oss | code | 0.133 | 0.867 | 0.0 |
| cg_oss | comment | 0.0 | 0.75 | 0.25 |
| cg_oss | direct | 0.0 | 0.383 | 0.617 |
| max_entropy | code | 0.167 | 0.8 | 0.033 |
| max_entropy | comment | 0.0 | 0.95 | 0.05 |
| max_entropy | direct | 0.0 | 0.567 | 0.433 |
| npo | code | 0.0 | 0.0 | 1.0 |
| npo | comment | 0.0 | 0.0 | 1.0 |
| npo | direct | 0.0 | 0.0 | 1.0 |

## Specific-correct rate per route (raw correct in parentheses)

| model | qa_zero | qa_para | qa_one | dict | assert | func | comment |
|---|---|---|---|---|---|---|---|
| base | 0.90 (0.90) | 0.95 (0.95) | 1.00 (1.00) | 0.85 (0.85) | 0.90 (0.90) | 1.00 (1.00) | 0.90 (0.95) |
| cg_oss | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 0.10 (0.15) | 0.20 (0.25) | 0.00 (0.00) | 0.00 (0.00) |
| max_entropy | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 0.15 (0.20) | 0.25 (0.30) | 0.00 (0.00) | 0.00 (0.00) |
| npo | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) |

## Cross-subject guess rate (answer matches another subject's true answer; chance-level control)

| model | code | direct |
|---|---|---|
| base | 0.0 | 0.0 |
| cg_oss | 0.017 | 0.0 |
| max_entropy | 0.012 | 0.0 |
| npo | 0.0 | 0.0 |

## Fact-level direct x code, on the 20 facts base Qwen knows through both routes

`code_only` = every direct route wrong but at least one code route specifically correct (candidate recovery). C_rec = code_only / (facts with every direct route wrong). CI: paired bootstrap over facts of (code accuracy - direct accuracy).

| model | facts | both | code_only | direct_only | neither | C_rec | mcnemar_p | code_minus_direct | ci95 |
|---|---|---|---|---|---|---|---|---|---|
| base | 20 | 20 | 0 | 0 | 0 | 0.0 | nan | -0.033 | [-0.133, 0.067] |
| cg_oss | 20 | 0 | 5 | 0 | 15 | 0.25 | 0.0625 | 0.1 | [0.033, 0.183] |
| max_entropy | 20 | 0 | 6 | 0 | 14 | 0.3 | 0.0312 | 0.133 | [0.050, 0.233] |
| npo | 20 | 0 | 0 | 0 | 20 | 0.0 | nan | 0.0 | [0.000, 0.000] |

## Candidate recoveries by relation (count of facts)

| model | author | capital | currency | language | symbol |
|---|---|---|---|---|---|
| base | 0 | 0 | 0 | 0 | 0 |
| cg_oss | 1 | 1 | 0 | 1 | 2 |
| max_entropy | 1 | 2 | 0 | 1 | 2 |
| npo | 0 | 0 | 0 | 0 | 0 |

## Every code-route recovery (direct routes all wrong)

| model | route | subject | span |
|---|---|---|---|
| cg_oss | assert | Japan | Tokyo |
| cg_oss | assert | Romeo and Juliet | William Shakespeare |
| cg_oss | assert | iron | Fe |
| cg_oss | dict | sodium | Na |
| cg_oss | assert | sodium | Na |
| cg_oss | dict | Mexico | Spanish |
| max_entropy | dict | Japan | Tokyo |
| max_entropy | assert | Japan | Tokyo |
| max_entropy | assert | Egypt | cairo |
| max_entropy | assert | Romeo and Juliet | William Shakespeare |
| max_entropy | assert | iron | Fe |
| max_entropy | dict | sodium | Na |
| max_entropy | assert | sodium | Na |
| max_entropy | dict | Mexico | Spanish |
