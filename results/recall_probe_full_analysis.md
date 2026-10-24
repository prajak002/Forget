# Cross-route recall probe: `full` (100 facts, models: base, cg_oss, max_entropy, npo)

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

## Specific-correct rate per route (raw correct in parentheses)

| model | qa_zero | qa_para | qa_one | dict | assert | func | comment |
|---|---|---|---|---|---|---|---|
| base | 0.92 (0.93) | 0.95 (0.95) | 0.93 (0.95) | 0.94 (0.95) | 0.88 (0.89) | 0.94 (0.94) | 0.72 (0.89) |
| cg_oss | 0.00 (0.00) | 0.00 (0.01) | 0.00 (0.00) | 0.19 (0.23) | 0.30 (0.34) | 0.00 (0.00) | 0.00 (0.00) |
| max_entropy | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 0.18 (0.21) | 0.27 (0.31) | 0.01 (0.01) | 0.00 (0.00) |
| npo | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) |

## Cross-subject guess rate (answer matches another subject's true answer; chance-level control)

| model | code | direct |
|---|---|---|
| base | 0.0 | 0.001 |
| cg_oss | 0.004 | 0.003 |
| max_entropy | 0.005 | 0.0 |
| npo | 0.0 | 0.0 |

## Fact-level direct x code, on the 96 facts base Qwen knows through both routes

`code_only` = every direct route wrong but at least one code route specifically correct (candidate recovery). C_rec = code_only / (facts with every direct route wrong). CI: paired bootstrap over facts of (code accuracy - direct accuracy). `non_derivable` drops facts whose answer can be spelled from the subject (helium -> He, Germany -> German), which need no world knowledge.

| subset | model | facts | both | code_only | direct_only | neither | C_rec | mcnemar_p | code_minus_direct | ci95 |
|---|---|---|---|---|---|---|---|---|---|---|
| all | base | 96 | 96 | 0 | 0 | 0 | 0.0 | nan | -0.014 | [-0.052, 0.024] |
| all | cg_oss | 96 | 0 | 32 | 0 | 64 | 0.333 | 4.7e-10 | 0.17 | [0.122, 0.222] |
| all | max_entropy | 96 | 0 | 30 | 0 | 66 | 0.312 | 1.9e-09 | 0.16 | [0.111, 0.212] |
| all | npo | 96 | 0 | 0 | 0 | 96 | 0.0 | nan | 0.0 | [0.000, 0.000] |
| non_derivable | base | 72 | 72 | 0 | 0 | 0 | 0.0 | nan | -0.014 | [-0.065, 0.037] |
| non_derivable | cg_oss | 72 | 0 | 11 | 0 | 61 | 0.153 | 0.00098 | 0.069 | [0.032, 0.111] |
| non_derivable | max_entropy | 72 | 0 | 10 | 0 | 62 | 0.139 | 0.002 | 0.069 | [0.032, 0.116] |
| non_derivable | npo | 72 | 0 | 0 | 0 | 72 | 0.0 | nan | 0.0 | [0.000, 0.000] |
| derivable | base | 24 | 24 | 0 | 0 | 0 | 0.0 | nan | -0.014 | [-0.042, 0.000] |
| derivable | cg_oss | 24 | 0 | 21 | 0 | 3 | 0.875 | 9.5e-07 | 0.472 | [0.375, 0.556] |
| derivable | max_entropy | 24 | 0 | 20 | 0 | 4 | 0.833 | 1.9e-06 | 0.431 | [0.333, 0.528] |
| derivable | npo | 24 | 0 | 0 | 0 | 24 | 0.0 | nan | 0.0 | [0.000, 0.000] |

## Candidate recoveries by relation (count of facts)

| model | author | capital | currency | language | symbol |
|---|---|---|---|---|---|
| base | 0 | 0 | 0 | 0 | 0 |
| cg_oss | 1 | 3 | 0 | 12 | 16 |
| max_entropy | 0 | 4 | 0 | 11 | 15 |
| npo | 0 | 0 | 0 | 0 | 0 |

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
