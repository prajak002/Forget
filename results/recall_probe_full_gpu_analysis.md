# Cross-route recall probe: `full_gpu` (100 facts, models: base, cg_oss)

## Response categories (share of responses)

| model | group | correct | fluent_incorrect | garbage |
|---|---|---|---|---|
| base | code | 0.923 | 0.077 | 0.0 |
| base | comment | 0.91 | 0.07 | 0.02 |
| base | direct | 0.933 | 0.067 | 0.0 |
| cg_oss | code | 0.19 | 0.797 | 0.013 |
| cg_oss | comment | 0.0 | 0.59 | 0.41 |
| cg_oss | direct | 0.003 | 0.28 | 0.717 |

## Specific-correct rate per route (raw correct in parentheses)

| model | qa_zero | qa_para | qa_one | dict | assert | func | comment |
|---|---|---|---|---|---|---|---|
| base | 0.90 (0.91) | 0.94 (0.94) | 0.93 (0.95) | 0.94 (0.95) | 0.86 (0.88) | 0.94 (0.94) | 0.74 (0.91) |
| cg_oss | 0.00 (0.00) | 0.00 (0.01) | 0.00 (0.00) | 0.19 (0.23) | 0.32 (0.34) | 0.00 (0.00) | 0.00 (0.00) |

## Cross-subject guess rate (answer matches another subject's true answer; chance-level control)

| model | code | direct |
|---|---|---|
| base | 0.001 | 0.001 |
| cg_oss | 0.004 | 0.003 |

## Fact-level direct x code, on the 95 facts base Qwen knows through both routes

`code_only` = every direct route wrong but at least one code route specifically correct (candidate recovery). C_rec = code_only / (facts with every direct route wrong). CI: paired bootstrap over facts of (code accuracy - direct accuracy). `non_derivable` drops facts whose answer can be spelled from the subject (helium -> He, Germany -> German), which need no world knowledge.

| subset | model | facts | both | code_only | direct_only | neither | C_rec | mcnemar_p | code_minus_direct | ci95 |
|---|---|---|---|---|---|---|---|---|---|---|
| all | base | 95 | 95 | 0 | 0 | 0 | 0.0 | nan | 0.0 | [-0.042, 0.042] |
| all | cg_oss | 95 | 0 | 33 | 0 | 62 | 0.347 | 2.3e-10 | 0.179 | [0.126, 0.235] |
| non_derivable | base | 71 | 71 | 0 | 0 | 0 | 0.0 | nan | 0.005 | [-0.052, 0.061] |
| non_derivable | cg_oss | 71 | 0 | 12 | 0 | 59 | 0.169 | 0.00049 | 0.075 | [0.038, 0.117] |
| derivable | base | 24 | 24 | 0 | 0 | 0 | 0.0 | nan | -0.014 | [-0.042, 0.000] |
| derivable | cg_oss | 24 | 0 | 21 | 0 | 3 | 0.875 | 9.5e-07 | 0.486 | [0.389, 0.569] |

## Candidate recoveries by relation (count of facts)

| model | author | capital | currency | language | symbol |
|---|---|---|---|---|---|
| base | 0 | 0 | 0 | 0 | 0 |
| cg_oss | 1 | 3 | 0 | 12 | 17 |

## Every code-route recovery (direct routes all wrong)

| model | route | subject | span | derivable |
|---|---|---|---|---|
| cg_oss | assert | France | Paris | False |
| cg_oss | assert | Japan | Tokyo | False |
| cg_oss | dict | Russia | Moscow | False |
| cg_oss | assert | Russia | Moscow | False |
| cg_oss | assert | Romeo and Juliet | William Shakespeare | False |
| cg_oss | dict | iron | Fe | False |
| cg_oss | assert | iron | Fe | False |
| cg_oss | dict | sodium | Na | False |
| cg_oss | assert | sodium | Na | False |
| cg_oss | assert | silver | Ag | False |
| cg_oss | dict | potassium | K | False |
| cg_oss | assert | potassium | K | False |
| cg_oss | assert | copper | Cu | False |
| cg_oss | assert | mercury | Hg | False |
| cg_oss | dict | helium | He | True |
| cg_oss | assert | helium | He | True |
| cg_oss | dict | neon | Ne | True |
| cg_oss | assert | neon | Ne | True |
| cg_oss | dict | chlorine | Cl | True |
| cg_oss | assert | chlorine | Cl | True |
| cg_oss | assert | magnesium | Mg | True |
| cg_oss | dict | zinc | Zn | True |
| cg_oss | assert | zinc | Zn | True |
| cg_oss | dict | aluminum | Al | True |
| cg_oss | assert | aluminum | Al | True |
| cg_oss | dict | nickel | Ni | True |
| cg_oss | assert | nickel | Ni | True |
| cg_oss | dict | sulfur | S | True |
| cg_oss | assert | sulfur | S | True |
| cg_oss | dict | nitrogen | N | True |
| cg_oss | assert | nitrogen | N | True |
| cg_oss | dict | hydrogen | H | True |
| cg_oss | assert | hydrogen | H | True |
| cg_oss | assert | tungsten | W | False |
| cg_oss | dict | Mexico | Spanish | False |
| cg_oss | dict | Germany | German | True |
| cg_oss | assert | Germany | German | True |
| cg_oss | dict | Greece | Greek | True |
| cg_oss | assert | Greece | Greek | True |
| cg_oss | assert | Turkey | Turkish | True |
| cg_oss | assert | Japan | Japanese | True |
| cg_oss | dict | Russia | Russian | True |
| cg_oss | assert | Russia | Russian | True |
| cg_oss | assert | Thailand | Thai | True |
| cg_oss | dict | Vietnam | Vietnamese | True |
| cg_oss | assert | Vietnam | Vietnamese | True |
| cg_oss | assert | Poland | Polish | True |
| cg_oss | assert | Sweden | Swedish | True |
| cg_oss | assert | Hungary | Hungarian | True |
| cg_oss | dict | Italy | Italian | True |
| cg_oss | assert | Italy | Italian | True |

## Facts per relation: written, known by base (direct and code), and not spellable

| relation | written | base_known | base_known_non_derivable |
|---|---|---|---|
| author | 20 | 19 | 19 |
| capital | 20 | 20 | 20 |
| currency | 20 | 16 | 16 |
| language | 20 | 20 | 6 |
| symbol | 20 | 20 | 10 |
| ALL | 100 | 95 | 71 |

## Recovery by relation, `non_derivable` facts (C_rec with Wilson 95% interval)

C_rec = facts recovered through at least one code route / facts every direct route failed. `chance_expected`: recoveries expected if code answers were guesses from the relation's answer pool (from the cross-subject guess rate). McNemar p per relation, Holm-corrected across relations.

| model | relation | facts | direct_failed | recovered | C_rec_ci | chance_expected | mcnemar_p | p_holm |
|---|---|---|---|---|---|---|---|---|
| cg_oss | author | 19 | 19 | 1 | 0.05 [0.01, 0.25] | 0.41 | 1.0 | 1.0 |
| cg_oss | capital | 20 | 20 | 3 | 0.15 [0.05, 0.36] | 0.31 | 0.25 | 0.75 |
| cg_oss | currency | 16 | 16 | 0 | 0.00 [0.00, 0.19] | 0.0 | nan | nan |
| cg_oss | language | 6 | 6 | 1 | 0.17 [0.03, 0.56] | 0.0 | 1.0 | 1.0 |
| cg_oss | symbol | 10 | 10 | 7 | 0.70 [0.40, 0.89] | 0.1 | 0.016 | 0.062 |
| cg_oss | ALL | 71 | 71 | 12 | 0.17 [0.10, 0.27] | 0.83 | 0.00049 | nan |

## Recovery by relation and route, `non_derivable` facts (rate [Wilson 95%] among direct-failed facts)

| model | relation | dict | assert | func | comment |
|---|---|---|---|---|---|
| cg_oss | ALL | 0.07 [0.03, 0.15] | 0.15 [0.09, 0.26] | 0.00 [0.00, 0.05] | 0.00 [0.00, 0.05] |
| cg_oss | author | 0.00 [0.00, 0.17] | 0.05 [0.01, 0.25] | 0.00 [0.00, 0.17] | 0.00 [0.00, 0.17] |
| cg_oss | capital | 0.05 [0.01, 0.24] | 0.15 [0.05, 0.36] | 0.00 [0.00, 0.16] | 0.00 [0.00, 0.16] |
| cg_oss | currency | 0.00 [0.00, 0.19] | 0.00 [0.00, 0.19] | 0.00 [0.00, 0.19] | 0.00 [0.00, 0.19] |
| cg_oss | language | 0.17 [0.03, 0.56] | 0.00 [0.00, 0.39] | 0.00 [0.00, 0.39] | 0.00 [0.00, 0.39] |
| cg_oss | symbol | 0.30 [0.11, 0.60] | 0.70 [0.40, 0.89] | 0.00 [0.00, 0.28] | 0.00 [0.00, 0.28] |

## Recovery by relation, `all` facts (C_rec with Wilson 95% interval)

C_rec = facts recovered through at least one code route / facts every direct route failed. `chance_expected`: recoveries expected if code answers were guesses from the relation's answer pool (from the cross-subject guess rate). McNemar p per relation, Holm-corrected across relations.

| model | relation | facts | direct_failed | recovered | C_rec_ci | chance_expected | mcnemar_p | p_holm |
|---|---|---|---|---|---|---|---|---|
| cg_oss | author | 19 | 19 | 1 | 0.05 [0.01, 0.25] | 0.41 | 1.0 | 1.0 |
| cg_oss | capital | 20 | 20 | 3 | 0.15 [0.05, 0.36] | 0.31 | 0.25 | 0.5 |
| cg_oss | currency | 16 | 16 | 0 | 0.00 [0.00, 0.19] | 0.0 | nan | nan |
| cg_oss | language | 20 | 20 | 12 | 0.60 [0.39, 0.78] | 0.15 | 0.00049 | 0.0015 |
| cg_oss | symbol | 20 | 20 | 17 | 0.85 [0.64, 0.95] | 0.1 | 1.5e-05 | 6.1e-05 |
| cg_oss | ALL | 95 | 95 | 33 | 0.35 [0.26, 0.45] | 0.98 | 2.3e-10 | nan |

## Recovery by relation and route, `all` facts (rate [Wilson 95%] among direct-failed facts)

| model | relation | dict | assert | func | comment |
|---|---|---|---|---|---|
| cg_oss | ALL | 0.20 [0.13, 0.29] | 0.34 [0.25, 0.44] | 0.00 [0.00, 0.04] | 0.00 [0.00, 0.04] |
| cg_oss | author | 0.00 [0.00, 0.17] | 0.05 [0.01, 0.25] | 0.00 [0.00, 0.17] | 0.00 [0.00, 0.17] |
| cg_oss | capital | 0.05 [0.01, 0.24] | 0.15 [0.05, 0.36] | 0.00 [0.00, 0.16] | 0.00 [0.00, 0.16] |
| cg_oss | currency | 0.00 [0.00, 0.19] | 0.00 [0.00, 0.19] | 0.00 [0.00, 0.19] | 0.00 [0.00, 0.19] |
| cg_oss | language | 0.30 [0.14, 0.52] | 0.55 [0.34, 0.74] | 0.00 [0.00, 0.16] | 0.00 [0.00, 0.16] |
| cg_oss | symbol | 0.60 [0.39, 0.78] | 0.85 [0.64, 0.95] | 0.00 [0.00, 0.16] | 0.00 [0.00, 0.16] |
