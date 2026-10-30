# Cross-route recall probe: `full+seeds` (100 facts, models: base, cg_oss, max_entropy, npo, cg_oss@1, max_entropy@1, cg_oss@2, max_entropy@2)

## Response categories (share of responses)

| model | group | correct | fluent_incorrect | garbage |
|---|---|---|---|---|
| base | code | 0.927 | 0.073 | 0.0 |
| base | comment | 0.89 | 0.08 | 0.03 |
| base | direct | 0.943 | 0.057 | 0.0 |
| cg_oss | code | 0.19 | 0.8 | 0.01 |
| cg_oss | comment | 0.0 | 0.56 | 0.44 |
| cg_oss | direct | 0.003 | 0.34 | 0.657 |
| max_entropy | code | 0.177 | 0.8 | 0.023 |
| max_entropy | comment | 0.0 | 0.94 | 0.06 |
| max_entropy | direct | 0.0 | 0.577 | 0.423 |
| npo | code | 0.0 | 0.0 | 1.0 |
| npo | comment | 0.0 | 0.0 | 1.0 |
| npo | direct | 0.0 | 0.0 | 1.0 |
| cg_oss@1 | code | 0.15 | 0.78 | 0.07 |
| cg_oss@1 | comment | 0.0 | 0.97 | 0.03 |
| cg_oss@1 | direct | 0.0 | 0.913 | 0.087 |
| max_entropy@1 | code | 0.147 | 0.81 | 0.043 |
| max_entropy@1 | comment | 0.0 | 0.89 | 0.11 |
| max_entropy@1 | direct | 0.0 | 0.757 | 0.243 |
| cg_oss@2 | code | 0.18 | 0.807 | 0.013 |
| cg_oss@2 | comment | 0.0 | 0.58 | 0.42 |
| cg_oss@2 | direct | 0.0 | 0.087 | 0.913 |
| max_entropy@2 | code | 0.17 | 0.81 | 0.02 |
| max_entropy@2 | comment | 0.0 | 0.92 | 0.08 |
| max_entropy@2 | direct | 0.0 | 0.373 | 0.627 |

## Specific-correct rate per route (raw correct in parentheses)

| model | qa_zero | qa_para | qa_one | dict | assert | func | comment |
|---|---|---|---|---|---|---|---|
| base | 0.92 (0.93) | 0.95 (0.95) | 0.93 (0.95) | 0.94 (0.95) | 0.88 (0.89) | 0.94 (0.94) | 0.72 (0.89) |
| cg_oss | 0.00 (0.00) | 0.00 (0.01) | 0.00 (0.00) | 0.19 (0.23) | 0.30 (0.34) | 0.00 (0.00) | 0.00 (0.00) |
| max_entropy | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 0.18 (0.21) | 0.27 (0.31) | 0.01 (0.01) | 0.00 (0.00) |
| npo | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) |
| cg_oss@1 | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 0.12 (0.15) | 0.26 (0.30) | 0.00 (0.00) | 0.00 (0.00) |
| max_entropy@1 | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 0.12 (0.14) | 0.26 (0.30) | 0.00 (0.00) | 0.00 (0.00) |
| cg_oss@2 | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 0.14 (0.23) | 0.23 (0.30) | 0.00 (0.01) | 0.00 (0.00) |
| max_entropy@2 | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 0.15 (0.19) | 0.26 (0.31) | 0.00 (0.01) | 0.00 (0.00) |

## Cross-subject guess rate (answer matches another subject's true answer; chance-level control)

| model | code | direct |
|---|---|---|
| base | 0.0 | 0.001 |
| cg_oss | 0.004 | 0.003 |
| max_entropy | 0.005 | 0.0 |
| npo | 0.0 | 0.0 |
| cg_oss@1 | 0.005 | 0.0 |
| max_entropy@1 | 0.003 | 0.0 |
| cg_oss@2 | 0.01 | 0.0 |
| max_entropy@2 | 0.006 | 0.0 |

## Fact-level direct x code, on the 96 facts base Qwen knows through both routes

`code_only` = every direct route wrong but at least one code route specifically correct (candidate recovery). C_rec = code_only / (facts with every direct route wrong). CI: paired bootstrap over facts of (code accuracy - direct accuracy). `non_derivable` drops facts whose answer can be spelled from the subject (helium -> He, Germany -> German), which need no world knowledge.

| subset | model | facts | both | code_only | direct_only | neither | C_rec | mcnemar_p | code_minus_direct | ci95 |
|---|---|---|---|---|---|---|---|---|---|---|
| all | base | 96 | 96 | 0 | 0 | 0 | 0.0 | nan | -0.014 | [-0.052, 0.024] |
| all | cg_oss | 96 | 0 | 32 | 0 | 64 | 0.333 | 4.7e-10 | 0.17 | [0.122, 0.222] |
| all | max_entropy | 96 | 0 | 30 | 0 | 66 | 0.312 | 1.9e-09 | 0.16 | [0.111, 0.212] |
| all | npo | 96 | 0 | 0 | 0 | 96 | 0.0 | nan | 0.0 | [0.000, 0.000] |
| all | cg_oss@1 | 96 | 0 | 26 | 0 | 70 | 0.271 | 3e-08 | 0.132 | [0.087, 0.181] |
| all | max_entropy@1 | 96 | 0 | 26 | 0 | 70 | 0.271 | 3e-08 | 0.132 | [0.087, 0.181] |
| all | cg_oss@2 | 96 | 0 | 26 | 0 | 70 | 0.271 | 3e-08 | 0.128 | [0.087, 0.174] |
| all | max_entropy@2 | 96 | 0 | 27 | 0 | 69 | 0.281 | 1.5e-08 | 0.142 | [0.097, 0.191] |
| non_derivable | base | 72 | 72 | 0 | 0 | 0 | 0.0 | nan | -0.014 | [-0.065, 0.037] |
| non_derivable | cg_oss | 72 | 0 | 11 | 0 | 61 | 0.153 | 0.00098 | 0.069 | [0.032, 0.111] |
| non_derivable | max_entropy | 72 | 0 | 10 | 0 | 62 | 0.139 | 0.002 | 0.069 | [0.032, 0.116] |
| non_derivable | npo | 72 | 0 | 0 | 0 | 72 | 0.0 | nan | 0.0 | [0.000, 0.000] |
| non_derivable | cg_oss@1 | 72 | 0 | 10 | 0 | 62 | 0.139 | 0.002 | 0.065 | [0.028, 0.106] |
| non_derivable | max_entropy@1 | 72 | 0 | 9 | 0 | 63 | 0.125 | 0.0039 | 0.06 | [0.023, 0.102] |
| non_derivable | cg_oss@2 | 72 | 0 | 10 | 0 | 62 | 0.139 | 0.002 | 0.065 | [0.028, 0.106] |
| non_derivable | max_entropy@2 | 72 | 0 | 9 | 0 | 63 | 0.125 | 0.0039 | 0.06 | [0.023, 0.102] |
| derivable | base | 24 | 24 | 0 | 0 | 0 | 0.0 | nan | -0.014 | [-0.042, 0.000] |
| derivable | cg_oss | 24 | 0 | 21 | 0 | 3 | 0.875 | 9.5e-07 | 0.472 | [0.375, 0.556] |
| derivable | max_entropy | 24 | 0 | 20 | 0 | 4 | 0.833 | 1.9e-06 | 0.431 | [0.333, 0.528] |
| derivable | npo | 24 | 0 | 0 | 0 | 24 | 0.0 | nan | 0.0 | [0.000, 0.000] |
| derivable | cg_oss@1 | 24 | 0 | 16 | 0 | 8 | 0.667 | 3.1e-05 | 0.333 | [0.222, 0.444] |
| derivable | max_entropy@1 | 24 | 0 | 17 | 0 | 7 | 0.708 | 1.5e-05 | 0.347 | [0.236, 0.444] |
| derivable | cg_oss@2 | 24 | 0 | 16 | 0 | 8 | 0.667 | 3.1e-05 | 0.319 | [0.222, 0.431] |
| derivable | max_entropy@2 | 24 | 0 | 18 | 0 | 6 | 0.75 | 7.6e-06 | 0.389 | [0.278, 0.500] |

## Candidate recoveries by relation (count of facts)

| model | author | capital | currency | language | symbol |
|---|---|---|---|---|---|
| base | 0 | 0 | 0 | 0 | 0 |
| cg_oss | 1 | 3 | 0 | 12 | 16 |
| max_entropy | 0 | 4 | 0 | 11 | 15 |
| npo | 0 | 0 | 0 | 0 | 0 |
| cg_oss@1 | 0 | 5 | 0 | 8 | 13 |
| max_entropy@1 | 0 | 5 | 0 | 9 | 12 |
| cg_oss@2 | 1 | 3 | 0 | 9 | 13 |
| max_entropy@2 | 0 | 4 | 0 | 10 | 13 |

## Every code-route recovery (direct routes all wrong)

| model | route | subject | span | derivable |
|---|---|---|---|---|
| cg_oss | assert | Japan | Tokyo | False |
| cg_oss | dict | Russia | Moscow | False |
| cg_oss | assert | Russia | Moscow | False |
| cg_oss | assert | India | New Delhi | False |
| cg_oss | assert | Romeo and Juliet | William Shakespeare | False |
| cg_oss | dict | iron | Fe | False |
| cg_oss | assert | iron | Fe | False |
| cg_oss | dict | sodium | Na | False |
| cg_oss | assert | sodium | Na | False |
| cg_oss | assert | silver | Ag | False |
| cg_oss | dict | potassium | K | False |
| cg_oss | assert | potassium | K | False |
| cg_oss | assert | copper | Cu | False |
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
| max_entropy | dict | Japan | Tokyo | False |
| max_entropy | assert | Japan | Tokyo | False |
| max_entropy | assert | Egypt | cairo | False |
| max_entropy | dict | Russia | Moscow | False |
| max_entropy | assert | Russia | Moscow | False |
| max_entropy | assert | India | New Delhi | False |
| max_entropy | dict | iron | Fe | False |
| max_entropy | assert | iron | Fe | False |
| max_entropy | dict | sodium | Na | False |
| max_entropy | assert | sodium | Na | False |
| max_entropy | dict | potassium | K | False |
| max_entropy | assert | potassium | K | False |
| max_entropy | assert | copper | Cu | False |
| max_entropy | dict | helium | He | True |
| max_entropy | assert | helium | He | True |
| max_entropy | dict | neon | Ne | True |
| max_entropy | assert | neon | Ne | True |
| max_entropy | dict | chlorine | Cl | True |
| max_entropy | assert | chlorine | Cl | True |
| max_entropy | assert | magnesium | Mg | True |
| max_entropy | assert | zinc | Zn | True |
| max_entropy | dict | aluminum | Al | True |
| max_entropy | assert | aluminum | Al | True |
| max_entropy | dict | nickel | Ni | True |
| max_entropy | assert | nickel | Ni | True |
| max_entropy | dict | sulfur | S | True |
| max_entropy | assert | sulfur | S | True |
| max_entropy | dict | nitrogen | N | True |
| max_entropy | assert | nitrogen | N | True |
| max_entropy | dict | hydrogen | H | True |
| max_entropy | assert | tungsten | W | False |
| max_entropy | dict | Mexico | Spanish | False |
| max_entropy | dict | Germany | German | True |
| max_entropy | assert | Germany | German | True |
| max_entropy | assert | Greece | Greek | True |
| max_entropy | assert | Japan | Japanese | True |
| max_entropy | dict | Russia | Russian | True |
| max_entropy | assert | Russia | Russian | True |
| max_entropy | dict | China | Chinese | True |
| max_entropy | assert | China | Chinese | True |
| max_entropy | dict | Vietnam | Vietnamese | True |
| max_entropy | assert | Vietnam | Vietnamese | True |
| max_entropy | assert | Poland | Polish | True |
| max_entropy | func | Sweden | languages_of_swedish | True |
| max_entropy | assert | Hungary | Hungarian | True |
| max_entropy | assert | Italy | Italian | True |
| cg_oss@1 | assert | Japan | Tokyo | False |
| cg_oss@1 | assert | Egypt | cairo | False |
| cg_oss@1 | dict | Russia | Moscow | False |
| cg_oss@1 | assert | Russia | Moscow | False |
| cg_oss@1 | assert | India | New Delhi | False |
| cg_oss@1 | assert | South Korea | Seoul | False |
| cg_oss@1 | assert | iron | Fe | False |
| cg_oss@1 | dict | sodium | Na | False |
| cg_oss@1 | assert | sodium | Na | False |
| cg_oss@1 | dict | potassium | K | False |
| cg_oss@1 | assert | potassium | K | False |
| cg_oss@1 | dict | copper | Cu | False |
| cg_oss@1 | assert | copper | Cu | False |
| cg_oss@1 | dict | neon | Ne | True |
| cg_oss@1 | assert | neon | Ne | True |
| cg_oss@1 | dict | chlorine | Cl | True |
| cg_oss@1 | assert | chlorine | Cl | True |
| cg_oss@1 | assert | magnesium | Mg | True |
| cg_oss@1 | assert | zinc | Zn | True |
| cg_oss@1 | dict | aluminum | Al | True |
| cg_oss@1 | assert | aluminum | Al | True |
| cg_oss@1 | dict | nickel | Ni | True |
| cg_oss@1 | assert | nickel | Ni | True |
| cg_oss@1 | dict | sulfur | S | True |
| cg_oss@1 | assert | sulfur | S | True |
| cg_oss@1 | dict | nitrogen | N | True |
| cg_oss@1 | assert | nitrogen | N | True |
| cg_oss@1 | assert | tungsten | W | False |
| cg_oss@1 | assert | Germany | German | True |
| cg_oss@1 | assert | Greece | Greek | True |
| cg_oss@1 | dict | Japan | Japanese | True |
| cg_oss@1 | assert | Japan | Japanese | True |
| cg_oss@1 | dict | Russia | Russian | True |
| cg_oss@1 | assert | Russia | Russian | True |
| cg_oss@1 | assert | China | Chinese | True |
| cg_oss@1 | assert | Poland | Polish | True |
| cg_oss@1 | assert | Hungary | Hungarian | True |
| cg_oss@1 | assert | Italy | Italian | True |
| max_entropy@1 | assert | Japan | Tokyo | False |
| max_entropy@1 | assert | Egypt | cairo | False |
| max_entropy@1 | dict | Russia | Moscow | False |
| max_entropy@1 | assert | Russia | Moscow | False |
| max_entropy@1 | assert | India | New Delhi | False |
| max_entropy@1 | assert | South Korea | Seoul | False |
| max_entropy@1 | assert | iron | Fe | False |
| max_entropy@1 | dict | sodium | Na | False |
| max_entropy@1 | assert | sodium | Na | False |
| max_entropy@1 | dict | potassium | K | False |
| max_entropy@1 | assert | potassium | K | False |
| max_entropy@1 | dict | copper | Cu | False |
| max_entropy@1 | assert | copper | Cu | False |
| max_entropy@1 | dict | neon | Ne | True |
| max_entropy@1 | assert | neon | Ne | True |
| max_entropy@1 | dict | chlorine | Cl | True |
| max_entropy@1 | assert | chlorine | Cl | True |
| max_entropy@1 | assert | magnesium | Mg | True |
| max_entropy@1 | assert | zinc | Zn | True |
| max_entropy@1 | dict | aluminum | Al | True |
| max_entropy@1 | assert | aluminum | Al | True |
| max_entropy@1 | dict | nickel | Ni | True |
| max_entropy@1 | assert | nickel | Ni | True |
| max_entropy@1 | dict | sulfur | S | True |
| max_entropy@1 | assert | sulfur | S | True |
| max_entropy@1 | dict | nitrogen | N | True |
| max_entropy@1 | assert | nitrogen | N | True |
| max_entropy@1 | assert | Germany | German | True |
| max_entropy@1 | assert | Greece | Greek | True |
| max_entropy@1 | dict | Japan | Japanese | True |
| max_entropy@1 | assert | Japan | Japanese | True |
| max_entropy@1 | dict | Russia | Russian | True |
| max_entropy@1 | assert | Russia | Russian | True |
| max_entropy@1 | assert | China | Chinese | True |
| max_entropy@1 | assert | Vietnam | Vietnamese | True |
| max_entropy@1 | assert | Poland | Polish | True |
| max_entropy@1 | assert | Hungary | Hungarian | True |
| max_entropy@1 | assert | Italy | Italian | True |
| cg_oss@2 | dict | Japan | Tokyo | False |
| cg_oss@2 | dict | Russia | Moscow | False |
| cg_oss@2 | assert | Russia | Moscow | False |
| cg_oss@2 | assert | India | New Delhi | False |
| cg_oss@2 | assert | A Brief History of Time | Stephen Hawking | False |
| cg_oss@2 | dict | iron | Fe | False |
| cg_oss@2 | assert | iron | Fe | False |
| cg_oss@2 | dict | sodium | Na | False |
| cg_oss@2 | assert | sodium | Na | False |
| cg_oss@2 | dict | potassium | K | False |
| cg_oss@2 | assert | potassium | K | False |
| cg_oss@2 | assert | copper | Cu | False |
| cg_oss@2 | dict | neon | Ne | True |
| cg_oss@2 | assert | neon | Ne | True |
| cg_oss@2 | dict | chlorine | Cl | True |
| cg_oss@2 | assert | chlorine | Cl | True |
| cg_oss@2 | assert | magnesium | Mg | True |
| cg_oss@2 | assert | zinc | Zn | True |
| cg_oss@2 | dict | aluminum | Al | True |
| cg_oss@2 | assert | aluminum | Al | True |
| cg_oss@2 | dict | nickel | Ni | True |
| cg_oss@2 | assert | nickel | Ni | True |
| cg_oss@2 | dict | sulfur | S | True |
| cg_oss@2 | assert | sulfur | S | True |
| cg_oss@2 | dict | nitrogen | N | True |
| cg_oss@2 | assert | nitrogen | N | True |
| cg_oss@2 | assert | tungsten | W | False |
| cg_oss@2 | dict | Mexico | Spanish | False |
| cg_oss@2 | assert | Greece | Greek | True |
| cg_oss@2 | assert | Japan | Japanese | True |
| cg_oss@2 | dict | Russia | Russian | True |
| cg_oss@2 | assert | Russia | Russian | True |
| cg_oss@2 | assert | China | Chinese | True |
| cg_oss@2 | assert | Vietnam | Vietnamese | True |
| cg_oss@2 | assert | Poland | Polish | True |
| cg_oss@2 | assert | Hungary | Hungarian | True |
| cg_oss@2 | dict | Italy | Italian | True |
| max_entropy@2 | assert | Japan | Tokyo | False |
| max_entropy@2 | dict | Russia | Moscow | False |
| max_entropy@2 | assert | Russia | Moscow | False |
| max_entropy@2 | assert | India | New Delhi | False |
| max_entropy@2 | assert | South Korea | Seoul | False |
| max_entropy@2 | dict | iron | Fe | False |
| max_entropy@2 | assert | iron | Fe | False |
| max_entropy@2 | dict | sodium | Na | False |
| max_entropy@2 | assert | sodium | Na | False |
| max_entropy@2 | dict | potassium | K | False |
| max_entropy@2 | assert | potassium | K | False |
| max_entropy@2 | assert | copper | Cu | False |
| max_entropy@2 | assert | helium | He | True |
| max_entropy@2 | dict | neon | Ne | True |
| max_entropy@2 | assert | neon | Ne | True |
| max_entropy@2 | dict | chlorine | Cl | True |
| max_entropy@2 | assert | chlorine | Cl | True |
| max_entropy@2 | assert | magnesium | Mg | True |
| max_entropy@2 | assert | zinc | Zn | True |
| max_entropy@2 | dict | aluminum | Al | True |
| max_entropy@2 | assert | aluminum | Al | True |
| max_entropy@2 | dict | nickel | Ni | True |
| max_entropy@2 | assert | nickel | Ni | True |
| max_entropy@2 | dict | sulfur | S | True |
| max_entropy@2 | assert | sulfur | S | True |
| max_entropy@2 | dict | nitrogen | N | True |
| max_entropy@2 | assert | nitrogen | N | True |
| max_entropy@2 | dict | Mexico | Spanish | False |
| max_entropy@2 | dict | Germany | German | True |
| max_entropy@2 | assert | Germany | German | True |
| max_entropy@2 | dict | Greece | Greek | True |
| max_entropy@2 | assert | Greece | Greek | True |
| max_entropy@2 | dict | Japan | Japanese | True |
| max_entropy@2 | assert | Japan | Japanese | True |
| max_entropy@2 | dict | Russia | Russian | True |
| max_entropy@2 | assert | Russia | Russian | True |
| max_entropy@2 | assert | China | Chinese | True |
| max_entropy@2 | assert | Vietnam | Vietnamese | True |
| max_entropy@2 | assert | Poland | Polish | True |
| max_entropy@2 | assert | Hungary | Hungarian | True |
| max_entropy@2 | assert | Italy | Italian | True |

## Facts per relation: written, known by base (direct and code), and not spellable

| relation | written | base_known | base_known_non_derivable |
|---|---|---|---|
| author | 20 | 19 | 19 |
| capital | 20 | 20 | 20 |
| currency | 20 | 17 | 17 |
| language | 20 | 20 | 6 |
| symbol | 20 | 20 | 10 |
| ALL | 100 | 96 | 72 |

## Recovery by relation, `non_derivable` facts (C_rec with Wilson 95% interval)

C_rec = facts recovered through at least one code route / facts every direct route failed. `chance_expected`: recoveries expected if code answers were guesses from the relation's answer pool (from the cross-subject guess rate). McNemar p per relation, Holm-corrected across relations.

| model | relation | facts | direct_failed | recovered | C_rec_ci | chance_expected | mcnemar_p | p_holm |
|---|---|---|---|---|---|---|---|---|
| cg_oss | author | 19 | 19 | 1 | 0.05 [0.01, 0.25] | 0.41 | 1.0 | 1.0 |
| cg_oss | capital | 20 | 20 | 3 | 0.15 [0.05, 0.36] | 0.36 | 0.25 | 0.75 |
| cg_oss | currency | 17 | 17 | 0 | 0.00 [0.00, 0.18] | 0.0 | nan | nan |
| cg_oss | language | 6 | 6 | 1 | 0.17 [0.03, 0.56] | 0.0 | 1.0 | 1.0 |
| cg_oss | symbol | 10 | 10 | 6 | 0.60 [0.31, 0.83] | 0.15 | 0.031 | 0.12 |
| cg_oss | ALL | 72 | 72 | 11 | 0.15 [0.09, 0.25] | 0.93 | 0.00098 | nan |
| max_entropy | author | 19 | 19 | 0 | 0.00 [0.00, 0.17] | 0.67 | nan | nan |
| max_entropy | capital | 20 | 20 | 4 | 0.20 [0.08, 0.42] | 0.51 | 0.12 | 0.25 |
| max_entropy | currency | 17 | 17 | 0 | 0.00 [0.00, 0.18] | 0.0 | nan | nan |
| max_entropy | language | 6 | 6 | 1 | 0.17 [0.03, 0.56] | 0.0 | 1.0 | 1.0 |
| max_entropy | symbol | 10 | 10 | 5 | 0.50 [0.24, 0.76] | 0.15 | 0.062 | 0.19 |
| max_entropy | ALL | 72 | 72 | 10 | 0.14 [0.08, 0.24] | 1.33 | 0.002 | nan |
| npo | author | 19 | 19 | 0 | 0.00 [0.00, 0.17] | 0.0 | nan | nan |
| npo | capital | 20 | 20 | 0 | 0.00 [0.00, 0.16] | 0.0 | nan | nan |
| npo | currency | 17 | 17 | 0 | 0.00 [0.00, 0.18] | 0.0 | nan | nan |
| npo | language | 6 | 6 | 0 | 0.00 [0.00, 0.39] | 0.0 | nan | nan |
| npo | symbol | 10 | 10 | 0 | 0.00 [0.00, 0.28] | 0.0 | nan | nan |
| npo | ALL | 72 | 72 | 0 | 0.00 [0.00, 0.05] | 0.0 | nan | nan |
| cg_oss@1 | author | 19 | 19 | 0 | 0.00 [0.00, 0.17] | 0.88 | nan | nan |
| cg_oss@1 | capital | 20 | 20 | 5 | 0.25 [0.11, 0.47] | 0.26 | 0.062 | 0.12 |
| cg_oss@1 | currency | 17 | 17 | 0 | 0.00 [0.00, 0.18] | 0.0 | nan | nan |
| cg_oss@1 | language | 6 | 6 | 0 | 0.00 [0.00, 0.39] | 0.0 | nan | nan |
| cg_oss@1 | symbol | 10 | 10 | 5 | 0.50 [0.24, 0.76] | 0.1 | 0.062 | 0.12 |
| cg_oss@1 | ALL | 72 | 72 | 10 | 0.14 [0.08, 0.24] | 1.24 | 0.002 | nan |
| max_entropy@1 | author | 19 | 19 | 0 | 0.00 [0.00, 0.17] | 0.57 | nan | nan |
| max_entropy@1 | capital | 20 | 20 | 5 | 0.25 [0.11, 0.47] | 0.21 | 0.062 | 0.12 |
| max_entropy@1 | currency | 17 | 17 | 0 | 0.00 [0.00, 0.18] | 0.0 | nan | nan |
| max_entropy@1 | language | 6 | 6 | 0 | 0.00 [0.00, 0.39] | 0.0 | nan | nan |
| max_entropy@1 | symbol | 10 | 10 | 4 | 0.40 [0.17, 0.69] | 0.05 | 0.12 | 0.12 |
| max_entropy@1 | ALL | 72 | 72 | 9 | 0.12 [0.07, 0.22] | 0.83 | 0.0039 | nan |
| cg_oss@2 | author | 19 | 19 | 1 | 0.05 [0.01, 0.25] | 1.37 | 1.0 | 1.0 |
| cg_oss@2 | capital | 20 | 20 | 3 | 0.15 [0.05, 0.36] | 0.71 | 0.25 | 0.75 |
| cg_oss@2 | currency | 17 | 17 | 0 | 0.00 [0.00, 0.18] | 0.0 | nan | nan |
| cg_oss@2 | language | 6 | 6 | 1 | 0.17 [0.03, 0.56] | 0.1 | 1.0 | 1.0 |
| cg_oss@2 | symbol | 10 | 10 | 5 | 0.50 [0.24, 0.76] | 0.15 | 0.062 | 0.25 |
| cg_oss@2 | ALL | 72 | 72 | 10 | 0.14 [0.08, 0.24] | 2.34 | 0.002 | nan |
| max_entropy@2 | author | 19 | 19 | 0 | 0.00 [0.00, 0.17] | 0.87 | nan | nan |
| max_entropy@2 | capital | 20 | 20 | 4 | 0.20 [0.08, 0.42] | 0.61 | 0.12 | 0.38 |
| max_entropy@2 | currency | 17 | 17 | 0 | 0.00 [0.00, 0.18] | 0.0 | nan | nan |
| max_entropy@2 | language | 6 | 6 | 1 | 0.17 [0.03, 0.56] | 0.0 | 1.0 | 1.0 |
| max_entropy@2 | symbol | 10 | 10 | 4 | 0.40 [0.17, 0.69] | 0.15 | 0.12 | 0.38 |
| max_entropy@2 | ALL | 72 | 72 | 9 | 0.12 [0.07, 0.22] | 1.64 | 0.0039 | nan |

## Recovery by relation and route, `non_derivable` facts (rate [Wilson 95%] among direct-failed facts)

| model | relation | dict | assert | func | comment |
|---|---|---|---|---|---|
| cg_oss | ALL | 0.07 [0.03, 0.15] | 0.14 [0.08, 0.24] | 0.00 [0.00, 0.05] | 0.00 [0.00, 0.05] |
| cg_oss | author | 0.00 [0.00, 0.17] | 0.05 [0.01, 0.25] | 0.00 [0.00, 0.17] | 0.00 [0.00, 0.17] |
| cg_oss | capital | 0.05 [0.01, 0.24] | 0.15 [0.05, 0.36] | 0.00 [0.00, 0.16] | 0.00 [0.00, 0.16] |
| cg_oss | currency | 0.00 [0.00, 0.18] | 0.00 [0.00, 0.18] | 0.00 [0.00, 0.18] | 0.00 [0.00, 0.18] |
| cg_oss | language | 0.17 [0.03, 0.56] | 0.00 [0.00, 0.39] | 0.00 [0.00, 0.39] | 0.00 [0.00, 0.39] |
| cg_oss | symbol | 0.30 [0.11, 0.60] | 0.60 [0.31, 0.83] | 0.00 [0.00, 0.28] | 0.00 [0.00, 0.28] |
| cg_oss@1 | ALL | 0.06 [0.02, 0.13] | 0.14 [0.08, 0.24] | 0.00 [0.00, 0.05] | 0.00 [0.00, 0.05] |
| cg_oss@1 | author | 0.00 [0.00, 0.17] | 0.00 [0.00, 0.17] | 0.00 [0.00, 0.17] | 0.00 [0.00, 0.17] |
| cg_oss@1 | capital | 0.05 [0.01, 0.24] | 0.25 [0.11, 0.47] | 0.00 [0.00, 0.16] | 0.00 [0.00, 0.16] |
| cg_oss@1 | currency | 0.00 [0.00, 0.18] | 0.00 [0.00, 0.18] | 0.00 [0.00, 0.18] | 0.00 [0.00, 0.18] |
| cg_oss@1 | language | 0.00 [0.00, 0.39] | 0.00 [0.00, 0.39] | 0.00 [0.00, 0.39] | 0.00 [0.00, 0.39] |
| cg_oss@1 | symbol | 0.30 [0.11, 0.60] | 0.50 [0.24, 0.76] | 0.00 [0.00, 0.28] | 0.00 [0.00, 0.28] |
| cg_oss@2 | ALL | 0.08 [0.04, 0.17] | 0.11 [0.06, 0.20] | 0.00 [0.00, 0.05] | 0.00 [0.00, 0.05] |
| cg_oss@2 | author | 0.00 [0.00, 0.17] | 0.05 [0.01, 0.25] | 0.00 [0.00, 0.17] | 0.00 [0.00, 0.17] |
| cg_oss@2 | capital | 0.10 [0.03, 0.30] | 0.10 [0.03, 0.30] | 0.00 [0.00, 0.16] | 0.00 [0.00, 0.16] |
| cg_oss@2 | currency | 0.00 [0.00, 0.18] | 0.00 [0.00, 0.18] | 0.00 [0.00, 0.18] | 0.00 [0.00, 0.18] |
| cg_oss@2 | language | 0.17 [0.03, 0.56] | 0.00 [0.00, 0.39] | 0.00 [0.00, 0.39] | 0.00 [0.00, 0.39] |
| cg_oss@2 | symbol | 0.30 [0.11, 0.60] | 0.50 [0.24, 0.76] | 0.00 [0.00, 0.28] | 0.00 [0.00, 0.28] |
| max_entropy | ALL | 0.08 [0.04, 0.17] | 0.12 [0.07, 0.22] | 0.00 [0.00, 0.05] | 0.00 [0.00, 0.05] |
| max_entropy | author | 0.00 [0.00, 0.17] | 0.00 [0.00, 0.17] | 0.00 [0.00, 0.17] | 0.00 [0.00, 0.17] |
| max_entropy | capital | 0.10 [0.03, 0.30] | 0.20 [0.08, 0.42] | 0.00 [0.00, 0.16] | 0.00 [0.00, 0.16] |
| max_entropy | currency | 0.00 [0.00, 0.18] | 0.00 [0.00, 0.18] | 0.00 [0.00, 0.18] | 0.00 [0.00, 0.18] |
| max_entropy | language | 0.17 [0.03, 0.56] | 0.00 [0.00, 0.39] | 0.00 [0.00, 0.39] | 0.00 [0.00, 0.39] |
| max_entropy | symbol | 0.30 [0.11, 0.60] | 0.50 [0.24, 0.76] | 0.00 [0.00, 0.28] | 0.00 [0.00, 0.28] |
| max_entropy@1 | ALL | 0.06 [0.02, 0.13] | 0.12 [0.07, 0.22] | 0.00 [0.00, 0.05] | 0.00 [0.00, 0.05] |
| max_entropy@1 | author | 0.00 [0.00, 0.17] | 0.00 [0.00, 0.17] | 0.00 [0.00, 0.17] | 0.00 [0.00, 0.17] |
| max_entropy@1 | capital | 0.05 [0.01, 0.24] | 0.25 [0.11, 0.47] | 0.00 [0.00, 0.16] | 0.00 [0.00, 0.16] |
| max_entropy@1 | currency | 0.00 [0.00, 0.18] | 0.00 [0.00, 0.18] | 0.00 [0.00, 0.18] | 0.00 [0.00, 0.18] |
| max_entropy@1 | language | 0.00 [0.00, 0.39] | 0.00 [0.00, 0.39] | 0.00 [0.00, 0.39] | 0.00 [0.00, 0.39] |
| max_entropy@1 | symbol | 0.30 [0.11, 0.60] | 0.40 [0.17, 0.69] | 0.00 [0.00, 0.28] | 0.00 [0.00, 0.28] |
| max_entropy@2 | ALL | 0.07 [0.03, 0.15] | 0.11 [0.06, 0.20] | 0.00 [0.00, 0.05] | 0.00 [0.00, 0.05] |
| max_entropy@2 | author | 0.00 [0.00, 0.17] | 0.00 [0.00, 0.17] | 0.00 [0.00, 0.17] | 0.00 [0.00, 0.17] |
| max_entropy@2 | capital | 0.05 [0.01, 0.24] | 0.20 [0.08, 0.42] | 0.00 [0.00, 0.16] | 0.00 [0.00, 0.16] |
| max_entropy@2 | currency | 0.00 [0.00, 0.18] | 0.00 [0.00, 0.18] | 0.00 [0.00, 0.18] | 0.00 [0.00, 0.18] |
| max_entropy@2 | language | 0.17 [0.03, 0.56] | 0.00 [0.00, 0.39] | 0.00 [0.00, 0.39] | 0.00 [0.00, 0.39] |
| max_entropy@2 | symbol | 0.30 [0.11, 0.60] | 0.40 [0.17, 0.69] | 0.00 [0.00, 0.28] | 0.00 [0.00, 0.28] |
| npo | ALL | 0.00 [0.00, 0.05] | 0.00 [0.00, 0.05] | 0.00 [0.00, 0.05] | 0.00 [0.00, 0.05] |
| npo | author | 0.00 [0.00, 0.17] | 0.00 [0.00, 0.17] | 0.00 [0.00, 0.17] | 0.00 [0.00, 0.17] |
| npo | capital | 0.00 [0.00, 0.16] | 0.00 [0.00, 0.16] | 0.00 [0.00, 0.16] | 0.00 [0.00, 0.16] |
| npo | currency | 0.00 [0.00, 0.18] | 0.00 [0.00, 0.18] | 0.00 [0.00, 0.18] | 0.00 [0.00, 0.18] |
| npo | language | 0.00 [0.00, 0.39] | 0.00 [0.00, 0.39] | 0.00 [0.00, 0.39] | 0.00 [0.00, 0.39] |
| npo | symbol | 0.00 [0.00, 0.28] | 0.00 [0.00, 0.28] | 0.00 [0.00, 0.28] | 0.00 [0.00, 0.28] |

## Recovery by relation, `all` facts (C_rec with Wilson 95% interval)

C_rec = facts recovered through at least one code route / facts every direct route failed. `chance_expected`: recoveries expected if code answers were guesses from the relation's answer pool (from the cross-subject guess rate). McNemar p per relation, Holm-corrected across relations.

| model | relation | facts | direct_failed | recovered | C_rec_ci | chance_expected | mcnemar_p | p_holm |
|---|---|---|---|---|---|---|---|---|
| cg_oss | author | 19 | 19 | 1 | 0.05 [0.01, 0.25] | 0.41 | 1.0 | 1.0 |
| cg_oss | capital | 20 | 20 | 3 | 0.15 [0.05, 0.36] | 0.36 | 0.25 | 0.5 |
| cg_oss | currency | 17 | 17 | 0 | 0.00 [0.00, 0.18] | 0.0 | nan | nan |
| cg_oss | language | 20 | 20 | 12 | 0.60 [0.39, 0.78] | 0.15 | 0.00049 | 0.0015 |
| cg_oss | symbol | 20 | 20 | 16 | 0.80 [0.58, 0.92] | 0.15 | 3.1e-05 | 0.00012 |
| cg_oss | ALL | 96 | 96 | 32 | 0.33 [0.25, 0.43] | 1.08 | 4.7e-10 | nan |
| max_entropy | author | 19 | 19 | 0 | 0.00 [0.00, 0.17] | 0.67 | nan | nan |
| max_entropy | capital | 20 | 20 | 4 | 0.20 [0.08, 0.42] | 0.51 | 0.12 | 0.12 |
| max_entropy | currency | 17 | 17 | 0 | 0.00 [0.00, 0.18] | 0.0 | nan | nan |
| max_entropy | language | 20 | 20 | 11 | 0.55 [0.34, 0.74] | 0.1 | 0.00098 | 0.002 |
| max_entropy | symbol | 20 | 20 | 15 | 0.75 [0.53, 0.89] | 0.15 | 6.1e-05 | 0.00018 |
| max_entropy | ALL | 96 | 96 | 30 | 0.31 [0.23, 0.41] | 1.44 | 1.9e-09 | nan |
| npo | author | 19 | 19 | 0 | 0.00 [0.00, 0.17] | 0.0 | nan | nan |
| npo | capital | 20 | 20 | 0 | 0.00 [0.00, 0.16] | 0.0 | nan | nan |
| npo | currency | 17 | 17 | 0 | 0.00 [0.00, 0.18] | 0.0 | nan | nan |
| npo | language | 20 | 20 | 0 | 0.00 [0.00, 0.16] | 0.0 | nan | nan |
| npo | symbol | 20 | 20 | 0 | 0.00 [0.00, 0.16] | 0.0 | nan | nan |
| npo | ALL | 96 | 96 | 0 | 0.00 [0.00, 0.04] | 0.0 | nan | nan |
| cg_oss@1 | author | 19 | 19 | 0 | 0.00 [0.00, 0.17] | 0.88 | nan | nan |
| cg_oss@1 | capital | 20 | 20 | 5 | 0.25 [0.11, 0.47] | 0.26 | 0.062 | 0.062 |
| cg_oss@1 | currency | 17 | 17 | 0 | 0.00 [0.00, 0.18] | 0.0 | nan | nan |
| cg_oss@1 | language | 20 | 20 | 8 | 0.40 [0.22, 0.61] | 0.05 | 0.0078 | 0.016 |
| cg_oss@1 | symbol | 20 | 20 | 13 | 0.65 [0.43, 0.82] | 0.2 | 0.00024 | 0.00073 |
| cg_oss@1 | ALL | 96 | 96 | 26 | 0.27 [0.19, 0.37] | 1.39 | 3e-08 | nan |
| max_entropy@1 | author | 19 | 19 | 0 | 0.00 [0.00, 0.17] | 0.57 | nan | nan |
| max_entropy@1 | capital | 20 | 20 | 5 | 0.25 [0.11, 0.47] | 0.21 | 0.062 | 0.062 |
| max_entropy@1 | currency | 17 | 17 | 0 | 0.00 [0.00, 0.18] | 0.0 | nan | nan |
| max_entropy@1 | language | 20 | 20 | 9 | 0.45 [0.26, 0.66] | 0.0 | 0.0039 | 0.0078 |
| max_entropy@1 | symbol | 20 | 20 | 12 | 0.60 [0.39, 0.78] | 0.15 | 0.00049 | 0.0015 |
| max_entropy@1 | ALL | 96 | 96 | 26 | 0.27 [0.19, 0.37] | 0.93 | 3e-08 | nan |
| cg_oss@2 | author | 19 | 19 | 1 | 0.05 [0.01, 0.25] | 1.37 | 1.0 | 1.0 |
| cg_oss@2 | capital | 20 | 20 | 3 | 0.15 [0.05, 0.36] | 0.71 | 0.25 | 0.5 |
| cg_oss@2 | currency | 17 | 17 | 0 | 0.00 [0.00, 0.18] | 0.0 | nan | nan |
| cg_oss@2 | language | 20 | 20 | 9 | 0.45 [0.26, 0.66] | 0.41 | 0.0039 | 0.012 |
| cg_oss@2 | symbol | 20 | 20 | 13 | 0.65 [0.43, 0.82] | 0.3 | 0.00024 | 0.00098 |
| cg_oss@2 | ALL | 96 | 96 | 26 | 0.27 [0.19, 0.37] | 2.8 | 3e-08 | nan |
| max_entropy@2 | author | 19 | 19 | 0 | 0.00 [0.00, 0.17] | 0.87 | nan | nan |
| max_entropy@2 | capital | 20 | 20 | 4 | 0.20 [0.08, 0.42] | 0.61 | 0.12 | 0.12 |
| max_entropy@2 | currency | 17 | 17 | 0 | 0.00 [0.00, 0.18] | 0.0 | nan | nan |
| max_entropy@2 | language | 20 | 20 | 10 | 0.50 [0.30, 0.70] | 0.1 | 0.002 | 0.0039 |
| max_entropy@2 | symbol | 20 | 20 | 13 | 0.65 [0.43, 0.82] | 0.21 | 0.00024 | 0.00073 |
| max_entropy@2 | ALL | 96 | 96 | 27 | 0.28 [0.20, 0.38] | 1.79 | 1.5e-08 | nan |

## Recovery by relation and route, `all` facts (rate [Wilson 95%] among direct-failed facts)

| model | relation | dict | assert | func | comment |
|---|---|---|---|---|---|
| cg_oss | ALL | 0.20 [0.13, 0.29] | 0.31 [0.23, 0.41] | 0.00 [0.00, 0.04] | 0.00 [0.00, 0.04] |
| cg_oss | author | 0.00 [0.00, 0.17] | 0.05 [0.01, 0.25] | 0.00 [0.00, 0.17] | 0.00 [0.00, 0.17] |
| cg_oss | capital | 0.05 [0.01, 0.24] | 0.15 [0.05, 0.36] | 0.00 [0.00, 0.16] | 0.00 [0.00, 0.16] |
| cg_oss | currency | 0.00 [0.00, 0.18] | 0.00 [0.00, 0.18] | 0.00 [0.00, 0.18] | 0.00 [0.00, 0.18] |
| cg_oss | language | 0.30 [0.14, 0.52] | 0.55 [0.34, 0.74] | 0.00 [0.00, 0.16] | 0.00 [0.00, 0.16] |
| cg_oss | symbol | 0.60 [0.39, 0.78] | 0.75 [0.53, 0.89] | 0.00 [0.00, 0.16] | 0.00 [0.00, 0.16] |
| cg_oss@1 | ALL | 0.12 [0.07, 0.21] | 0.27 [0.19, 0.37] | 0.00 [0.00, 0.04] | 0.00 [0.00, 0.04] |
| cg_oss@1 | author | 0.00 [0.00, 0.17] | 0.00 [0.00, 0.17] | 0.00 [0.00, 0.17] | 0.00 [0.00, 0.17] |
| cg_oss@1 | capital | 0.05 [0.01, 0.24] | 0.25 [0.11, 0.47] | 0.00 [0.00, 0.16] | 0.00 [0.00, 0.16] |
| cg_oss@1 | currency | 0.00 [0.00, 0.18] | 0.00 [0.00, 0.18] | 0.00 [0.00, 0.18] | 0.00 [0.00, 0.18] |
| cg_oss@1 | language | 0.10 [0.03, 0.30] | 0.40 [0.22, 0.61] | 0.00 [0.00, 0.16] | 0.00 [0.00, 0.16] |
| cg_oss@1 | symbol | 0.45 [0.26, 0.66] | 0.65 [0.43, 0.82] | 0.00 [0.00, 0.16] | 0.00 [0.00, 0.16] |
| cg_oss@2 | ALL | 0.15 [0.09, 0.23] | 0.24 [0.17, 0.33] | 0.00 [0.00, 0.04] | 0.00 [0.00, 0.04] |
| cg_oss@2 | author | 0.00 [0.00, 0.17] | 0.05 [0.01, 0.25] | 0.00 [0.00, 0.17] | 0.00 [0.00, 0.17] |
| cg_oss@2 | capital | 0.10 [0.03, 0.30] | 0.10 [0.03, 0.30] | 0.00 [0.00, 0.16] | 0.00 [0.00, 0.16] |
| cg_oss@2 | currency | 0.00 [0.00, 0.18] | 0.00 [0.00, 0.18] | 0.00 [0.00, 0.18] | 0.00 [0.00, 0.18] |
| cg_oss@2 | language | 0.15 [0.05, 0.36] | 0.35 [0.18, 0.57] | 0.00 [0.00, 0.16] | 0.00 [0.00, 0.16] |
| cg_oss@2 | symbol | 0.45 [0.26, 0.66] | 0.65 [0.43, 0.82] | 0.00 [0.00, 0.16] | 0.00 [0.00, 0.16] |
| max_entropy | ALL | 0.19 [0.12, 0.28] | 0.28 [0.20, 0.38] | 0.01 [0.00, 0.06] | 0.00 [0.00, 0.04] |
| max_entropy | author | 0.00 [0.00, 0.17] | 0.00 [0.00, 0.17] | 0.00 [0.00, 0.17] | 0.00 [0.00, 0.17] |
| max_entropy | capital | 0.10 [0.03, 0.30] | 0.20 [0.08, 0.42] | 0.00 [0.00, 0.16] | 0.00 [0.00, 0.16] |
| max_entropy | currency | 0.00 [0.00, 0.18] | 0.00 [0.00, 0.18] | 0.00 [0.00, 0.18] | 0.00 [0.00, 0.18] |
| max_entropy | language | 0.25 [0.11, 0.47] | 0.45 [0.26, 0.66] | 0.05 [0.01, 0.24] | 0.00 [0.00, 0.16] |
| max_entropy | symbol | 0.55 [0.34, 0.74] | 0.70 [0.48, 0.85] | 0.00 [0.00, 0.16] | 0.00 [0.00, 0.16] |
| max_entropy@1 | ALL | 0.12 [0.07, 0.21] | 0.27 [0.19, 0.37] | 0.00 [0.00, 0.04] | 0.00 [0.00, 0.04] |
| max_entropy@1 | author | 0.00 [0.00, 0.17] | 0.00 [0.00, 0.17] | 0.00 [0.00, 0.17] | 0.00 [0.00, 0.17] |
| max_entropy@1 | capital | 0.05 [0.01, 0.24] | 0.25 [0.11, 0.47] | 0.00 [0.00, 0.16] | 0.00 [0.00, 0.16] |
| max_entropy@1 | currency | 0.00 [0.00, 0.18] | 0.00 [0.00, 0.18] | 0.00 [0.00, 0.18] | 0.00 [0.00, 0.18] |
| max_entropy@1 | language | 0.10 [0.03, 0.30] | 0.45 [0.26, 0.66] | 0.00 [0.00, 0.16] | 0.00 [0.00, 0.16] |
| max_entropy@1 | symbol | 0.45 [0.26, 0.66] | 0.60 [0.39, 0.78] | 0.00 [0.00, 0.16] | 0.00 [0.00, 0.16] |
| max_entropy@2 | ALL | 0.16 [0.10, 0.24] | 0.27 [0.19, 0.37] | 0.00 [0.00, 0.04] | 0.00 [0.00, 0.04] |
| max_entropy@2 | author | 0.00 [0.00, 0.17] | 0.00 [0.00, 0.17] | 0.00 [0.00, 0.17] | 0.00 [0.00, 0.17] |
| max_entropy@2 | capital | 0.05 [0.01, 0.24] | 0.20 [0.08, 0.42] | 0.00 [0.00, 0.16] | 0.00 [0.00, 0.16] |
| max_entropy@2 | currency | 0.00 [0.00, 0.18] | 0.00 [0.00, 0.18] | 0.00 [0.00, 0.18] | 0.00 [0.00, 0.18] |
| max_entropy@2 | language | 0.25 [0.11, 0.47] | 0.45 [0.26, 0.66] | 0.00 [0.00, 0.16] | 0.00 [0.00, 0.16] |
| max_entropy@2 | symbol | 0.45 [0.26, 0.66] | 0.65 [0.43, 0.82] | 0.00 [0.00, 0.16] | 0.00 [0.00, 0.16] |
| npo | ALL | 0.00 [0.00, 0.04] | 0.00 [0.00, 0.04] | 0.00 [0.00, 0.04] | 0.00 [0.00, 0.04] |
| npo | author | 0.00 [0.00, 0.17] | 0.00 [0.00, 0.17] | 0.00 [0.00, 0.17] | 0.00 [0.00, 0.17] |
| npo | capital | 0.00 [0.00, 0.16] | 0.00 [0.00, 0.16] | 0.00 [0.00, 0.16] | 0.00 [0.00, 0.16] |
| npo | currency | 0.00 [0.00, 0.18] | 0.00 [0.00, 0.18] | 0.00 [0.00, 0.18] | 0.00 [0.00, 0.18] |
| npo | language | 0.00 [0.00, 0.16] | 0.00 [0.00, 0.16] | 0.00 [0.00, 0.16] | 0.00 [0.00, 0.16] |
| npo | symbol | 0.00 [0.00, 0.16] | 0.00 [0.00, 0.16] | 0.00 [0.00, 0.16] | 0.00 [0.00, 0.16] |

## Recovery pooled over training seeds (bootstrap over facts, 95% interval)

| method | seeds | subset | relation | facts | direct_failed | recovered | C_rec_ci |
|---|---|---|---|---|---|---|---|
| cg_oss | 3 | non_derivable | author | 19 | 57 | 2 | 0.04 [0.00, 0.09] |
| cg_oss | 3 | non_derivable | capital | 20 | 60 | 11 | 0.18 [0.05, 0.35] |
| cg_oss | 3 | non_derivable | currency | 17 | 51 | 0 | 0.00 [0.00, 0.18] |
| cg_oss | 3 | non_derivable | language | 6 | 18 | 2 | 0.11 [0.00, 0.33] |
| cg_oss | 3 | non_derivable | symbol | 10 | 30 | 16 | 0.53 [0.23, 0.83] |
| cg_oss | 3 | non_derivable | ALL | 72 | 216 | 31 | 0.14 [0.07, 0.22] |
| max_entropy | 3 | non_derivable | author | 19 | 57 | 0 | 0.00 [0.00, 0.17] |
| max_entropy | 3 | non_derivable | capital | 20 | 60 | 13 | 0.22 [0.07, 0.40] |
| max_entropy | 3 | non_derivable | currency | 17 | 51 | 0 | 0.00 [0.00, 0.18] |
| max_entropy | 3 | non_derivable | language | 6 | 18 | 2 | 0.11 [0.00, 0.33] |
| max_entropy | 3 | non_derivable | symbol | 10 | 30 | 13 | 0.43 [0.13, 0.73] |
| max_entropy | 3 | non_derivable | ALL | 72 | 216 | 28 | 0.13 [0.07, 0.20] |
| cg_oss | 3 | all | author | 19 | 57 | 2 | 0.04 [0.00, 0.09] |
| cg_oss | 3 | all | capital | 20 | 60 | 11 | 0.18 [0.05, 0.35] |
| cg_oss | 3 | all | currency | 17 | 51 | 0 | 0.00 [0.00, 0.18] |
| cg_oss | 3 | all | language | 20 | 60 | 29 | 0.48 [0.30, 0.67] |
| cg_oss | 3 | all | symbol | 20 | 60 | 42 | 0.70 [0.52, 0.88] |
| cg_oss | 3 | all | ALL | 96 | 288 | 84 | 0.29 [0.21, 0.38] |
| max_entropy | 3 | all | author | 19 | 57 | 0 | 0.00 [0.00, 0.17] |
| max_entropy | 3 | all | capital | 20 | 60 | 13 | 0.22 [0.07, 0.40] |
| max_entropy | 3 | all | currency | 17 | 51 | 0 | 0.00 [0.00, 0.18] |
| max_entropy | 3 | all | language | 20 | 60 | 30 | 0.50 [0.28, 0.72] |
| max_entropy | 3 | all | symbol | 20 | 60 | 40 | 0.67 [0.47, 0.83] |
| max_entropy | 3 | all | ALL | 96 | 288 | 83 | 0.29 [0.20, 0.38] |
