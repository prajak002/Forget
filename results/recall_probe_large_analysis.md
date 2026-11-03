# Cross-route recall probe: `large` (895 facts, models: base, cg_oss)

## Response categories (share of responses)

| model | group | correct | fluent_incorrect | garbage |
|---|---|---|---|---|
| base | code | 0.564 | 0.435 | 0.001 |
| base | comment | 0.564 | 0.396 | 0.04 |
| base | direct | 0.649 | 0.351 | 0.0 |
| cg_oss | code | 0.07 | 0.917 | 0.012 |
| cg_oss | comment | 0.0 | 0.579 | 0.421 |
| cg_oss | direct | 0.0 | 0.295 | 0.705 |

## Specific-correct rate per route (raw correct in parentheses)

| model | qa_zero | qa_para | qa_one | dict | assert | func | comment |
|---|---|---|---|---|---|---|---|
| base | 0.58 (0.65) | 0.55 (0.63) | 0.59 (0.67) | 0.49 (0.57) | 0.47 (0.56) | 0.49 (0.56) | 0.41 (0.56) |
| cg_oss | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 0.05 (0.08) | 0.09 (0.12) | 0.00 (0.01) | 0.00 (0.00) |

## Cross-subject guess rate (answer matches another subject's true answer; chance-level control)

| model | code | direct |
|---|---|---|
| base | 0.003 | 0.002 |
| cg_oss | 0.003 | 0.0 |

## Fact-level direct x code, on the 519 facts base Qwen knows through both routes

`code_only` = every direct route wrong but at least one code route specifically correct (candidate recovery). C_rec = code_only / (facts with every direct route wrong). CI: paired bootstrap over facts of (code accuracy - direct accuracy). `non_derivable` drops facts whose answer can be spelled from the subject (helium -> He, Germany -> German), which need no world knowledge.

| subset | model | facts | both | code_only | direct_only | neither | C_rec | mcnemar_p | code_minus_direct | ci95 |
|---|---|---|---|---|---|---|---|---|---|---|
| all | base | 519 | 519 | 0 | 0 | 0 | 0.0 | nan | -0.058 | [-0.085, -0.031] |
| all | cg_oss | 519 | 0 | 85 | 0 | 434 | 0.164 | 5.2e-26 | 0.08 | [0.064, 0.098] |
| non_derivable | base | 412 | 412 | 0 | 0 | 0 | 0.0 | nan | -0.08 | [-0.109, -0.050] |
| non_derivable | cg_oss | 412 | 0 | 19 | 0 | 393 | 0.046 | 3.8e-06 | 0.018 | [0.011, 0.027] |
| derivable | base | 107 | 107 | 0 | 0 | 0 | 0.0 | nan | 0.028 | [-0.022, 0.078] |
| derivable | cg_oss | 107 | 0 | 66 | 0 | 41 | 0.617 | 2.7e-20 | 0.321 | [0.268, 0.377] |

## Candidate recoveries by relation (count of facts)

| model | atomic_number | author | capital | currency | founder | inventor | language | maker | river | sport | symbol | unit | year |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| cg_oss | 8 | 0 | 2 | 1 | 0 | 0 | 16 | 0 | 0 | 0 | 55 | 2 | 1 |

## Every code-route recovery (direct routes all wrong)

| model | route | subject | span | derivable |
|---|---|---|---|---|
| cg_oss | dict | iron | Fe | False |
| cg_oss | assert | iron | Fe | False |
| cg_oss | assert | sodium | Na | False |
| cg_oss | assert | silver | Ag | False |
| cg_oss | dict | potassium | K | False |
| cg_oss | assert | potassium | K | False |
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
| cg_oss | dict | sulfur | S | True |
| cg_oss | assert | sulfur | S | True |
| cg_oss | dict | nitrogen | N | True |
| cg_oss | assert | nitrogen | N | True |
| cg_oss | dict | hydrogen | H | True |
| cg_oss | assert | hydrogen | H | True |
| cg_oss | assert | tungsten | W | False |
| cg_oss | dict | Germany | German | True |
| cg_oss | assert | Germany | German | True |
| cg_oss | dict | Greece | Greek | True |
| cg_oss | assert | Japan | Japanese | True |
| cg_oss | assert | Russia | Russian | True |
| cg_oss | assert | Thailand | Thai | True |
| cg_oss | dict | Vietnam | Vietnamese | True |
| cg_oss | assert | Vietnam | Vietnamese | True |
| cg_oss | assert | Poland | Polish | True |
| cg_oss | assert | Sweden | Swedish | True |
| cg_oss | assert | Hungary | Hungarian | True |
| cg_oss | dict | Italy | Italian | True |
| cg_oss | assert | Italy | Italian | True |
| cg_oss | assert | Indonesia | Jakarta | False |
| cg_oss | assert | Taiwan | Taipei | True |
| cg_oss | dict | oxygen | O | True |
| cg_oss | assert | oxygen | O | True |
| cg_oss | func | oxygen | O | True |
| cg_oss | assert | carbon | C | True |
| cg_oss | dict | lithium | Li | True |
| cg_oss | assert | lithium | Li | True |
| cg_oss | dict | boron | B | True |
| cg_oss | assert | boron | B | True |
| cg_oss | dict | fluorine | F | True |
| cg_oss | assert | fluorine | F | True |
| cg_oss | assert | silicon | Si | True |
| cg_oss | assert | phosphorus | P | True |
| cg_oss | dict | argon | Ar | True |
| cg_oss | assert | argon | Ar | True |
| cg_oss | dict | scandium | Sc | True |
| cg_oss | assert | scandium | Sc | True |
| cg_oss | dict | vanadium | V | True |
| cg_oss | assert | vanadium | V | True |
| cg_oss | assert | manganese | Mn | True |
| cg_oss | dict | cobalt | Co | True |
| cg_oss | assert | cobalt | Co | True |
| cg_oss | assert | gallium | Ga | True |
| cg_oss | dict | selenium | Se | True |
| cg_oss | assert | selenium | Se | True |
| cg_oss | dict | bromine | Br | True |
| cg_oss | assert | bromine | Br | True |
| cg_oss | dict | krypton | Kr | True |
| cg_oss | assert | krypton | Kr | True |
| cg_oss | assert | rubidium | Rb | True |
| cg_oss | dict | strontium | Sr | True |
| cg_oss | assert | strontium | Sr | True |
| cg_oss | dict | yttrium | Y | True |
| cg_oss | assert | yttrium | Y | True |
| cg_oss | assert | zirconium | Zr | True |
| cg_oss | dict | molybdenum | Mo | True |
| cg_oss | assert | molybdenum | Mo | True |
| cg_oss | assert | ruthenium | Ru | True |
| cg_oss | dict | rhodium | Rh | True |
| cg_oss | assert | rhodium | Rh | True |
| cg_oss | dict | cadmium | Cd | True |
| cg_oss | assert | cadmium | Cd | True |
| cg_oss | assert | antimony | Sb | False |
| cg_oss | assert | tellurium | Te | True |
| cg_oss | dict | cesium | Cs | True |
| cg_oss | assert | cesium | Cs | True |
| cg_oss | assert | barium | Ba | True |
| cg_oss | dict | lanthanum | La | True |
| cg_oss | assert | lanthanum | La | True |
| cg_oss | dict | cerium | Ce | True |
| cg_oss | assert | cerium | Ce | True |
| cg_oss | dict | neodymium | Nd | True |
| cg_oss | assert | neodymium | Nd | True |
| cg_oss | dict | platinum | Pt | True |
| cg_oss | dict | osmium | Os | True |
| cg_oss | assert | osmium | Os | True |
| cg_oss | dict | iridium | Ir | True |
| cg_oss | assert | iridium | Ir | True |
| cg_oss | assert | bismuth | Bi | True |
| cg_oss | dict | polonium | Po | True |
| cg_oss | assert | polonium | Po | True |
| cg_oss | assert | thorium | Th | True |
| cg_oss | dict | uranium | U | True |
| cg_oss | assert | uranium | U | True |
| cg_oss | dict | plutonium | Pu | True |
| cg_oss | assert | plutonium | Pu | True |
| cg_oss | assert | thallium | Tl | True |
| cg_oss | func | Sierra Leone | currency_of_sierra_leone | True |
| cg_oss | assert | Norway | Norwegian | True |
| cg_oss | assert | Serbia | Serbian | True |
| cg_oss | assert | Armenia | Armenian | True |
| cg_oss | assert | Albania | Albanian | True |
| cg_oss | assert | Mongolia | Mongolian | True |
| cg_oss | assert | Malta | Maltese | True |
| cg_oss | assert | Uzbekistan | Uzbek | True |
| cg_oss | assert | hydrogen | 1 | False |
| cg_oss | assert | helium | 2 | False |
| cg_oss | assert | lithium | 3 | False |
| cg_oss | assert | beryllium | 4 | False |
| cg_oss | dict | boron | 5 | False |
| cg_oss | assert | boron | 5 | False |
| cg_oss | assert | sodium | 11 | False |
| cg_oss | assert | aluminum | 13 | False |
| cg_oss | dict | phosphorus | 15 | False |
| cg_oss | func | the start of the American Civil War | 1861 | False |
| cg_oss | dict | thermodynamic temperature | kelvin | False |
| cg_oss | dict | length | meter | False |

## Facts per relation: written, known by base (direct and code), and not spellable

| relation | written | base_known | base_known_non_derivable |
|---|---|---|---|
| atomic_number | 73 | 39 | 39 |
| author | 120 | 59 | 59 |
| capital | 132 | 128 | 123 |
| currency | 69 | 35 | 33 |
| founder | 52 | 2 | 2 |
| inventor | 75 | 12 | 9 |
| language | 62 | 54 | 16 |
| maker | 54 | 15 | 15 |
| river | 66 | 13 | 13 |
| sport | 19 | 6 | 5 |
| symbol | 69 | 69 | 11 |
| unit | 27 | 17 | 17 |
| year | 77 | 70 | 70 |
| ALL | 895 | 519 | 412 |

## Recovery by relation, `non_derivable` facts (C_rec with Wilson 95% interval)

C_rec = facts recovered through at least one code route / facts every direct route failed. `chance_expected`: recoveries expected if code answers were guesses from the relation's answer pool (from the cross-subject guess rate). McNemar p per relation, Holm-corrected across relations.

| model | relation | facts | direct_failed | recovered | C_rec_ci | chance_expected | mcnemar_p | p_holm |
|---|---|---|---|---|---|---|---|---|
| cg_oss | atomic_number | 39 | 39 | 8 | 0.20 [0.11, 0.35] | 0.99 | 0.0078 | 0.039 |
| cg_oss | author | 59 | 59 | 0 | 0.00 [0.00, 0.06] | 0.24 | nan | nan |
| cg_oss | capital | 123 | 123 | 1 | 0.01 [0.00, 0.04] | 0.47 | 1.0 | 1.0 |
| cg_oss | currency | 33 | 33 | 0 | 0.00 [0.00, 0.10] | 0.0 | nan | nan |
| cg_oss | founder | 2 | 2 | 0 | 0.00 [0.00, 0.66] | 0.0 | nan | nan |
| cg_oss | inventor | 9 | 9 | 0 | 0.00 [0.00, 0.30] | 0.0 | nan | nan |
| cg_oss | language | 16 | 16 | 0 | 0.00 [0.00, 0.19] | 0.08 | nan | nan |
| cg_oss | maker | 15 | 15 | 0 | 0.00 [0.00, 0.20] | 0.0 | nan | nan |
| cg_oss | river | 13 | 13 | 0 | 0.00 [0.00, 0.23] | 0.02 | nan | nan |
| cg_oss | sport | 5 | 5 | 0 | 0.00 [0.00, 0.43] | 0.48 | nan | nan |
| cg_oss | symbol | 11 | 11 | 7 | 0.64 [0.35, 0.85] | 0.09 | 0.016 | 0.062 |
| cg_oss | unit | 17 | 17 | 2 | 0.12 [0.03, 0.34] | 0.79 | 0.5 | 1.0 |
| cg_oss | year | 70 | 70 | 1 | 0.01 [0.00, 0.08] | 0.89 | 1.0 | 1.0 |
| cg_oss | ALL | 412 | 412 | 19 | 0.05 [0.03, 0.07] | 4.05 | 3.8e-06 | nan |

## Recovery by relation and route, `non_derivable` facts (rate [Wilson 95%] among direct-failed facts)

| model | relation | dict | assert | func | comment |
|---|---|---|---|---|---|
| cg_oss | ALL | 0.01 [0.01, 0.03] | 0.04 [0.02, 0.06] | 0.00 [0.00, 0.01] | 0.00 [0.00, 0.01] |
| cg_oss | atomic_number | 0.05 [0.01, 0.17] | 0.18 [0.09, 0.33] | 0.00 [0.00, 0.09] | 0.00 [0.00, 0.09] |
| cg_oss | author | 0.00 [0.00, 0.06] | 0.00 [0.00, 0.06] | 0.00 [0.00, 0.06] | 0.00 [0.00, 0.06] |
| cg_oss | capital | 0.00 [0.00, 0.03] | 0.01 [0.00, 0.04] | 0.00 [0.00, 0.03] | 0.00 [0.00, 0.03] |
| cg_oss | currency | 0.00 [0.00, 0.10] | 0.00 [0.00, 0.10] | 0.00 [0.00, 0.10] | 0.00 [0.00, 0.10] |
| cg_oss | founder | 0.00 [0.00, 0.66] | 0.00 [0.00, 0.66] | 0.00 [0.00, 0.66] | 0.00 [0.00, 0.66] |
| cg_oss | inventor | 0.00 [0.00, 0.30] | 0.00 [0.00, 0.30] | 0.00 [0.00, 0.30] | 0.00 [0.00, 0.30] |
| cg_oss | language | 0.00 [0.00, 0.19] | 0.00 [0.00, 0.19] | 0.00 [0.00, 0.19] | 0.00 [0.00, 0.19] |
| cg_oss | maker | 0.00 [0.00, 0.20] | 0.00 [0.00, 0.20] | 0.00 [0.00, 0.20] | 0.00 [0.00, 0.20] |
| cg_oss | river | 0.00 [0.00, 0.23] | 0.00 [0.00, 0.23] | 0.00 [0.00, 0.23] | 0.00 [0.00, 0.23] |
| cg_oss | sport | 0.00 [0.00, 0.43] | 0.00 [0.00, 0.43] | 0.00 [0.00, 0.43] | 0.00 [0.00, 0.43] |
| cg_oss | symbol | 0.18 [0.05, 0.48] | 0.64 [0.35, 0.85] | 0.00 [0.00, 0.26] | 0.00 [0.00, 0.26] |
| cg_oss | unit | 0.12 [0.03, 0.34] | 0.00 [0.00, 0.18] | 0.00 [0.00, 0.18] | 0.00 [0.00, 0.18] |
| cg_oss | year | 0.00 [0.00, 0.05] | 0.00 [0.00, 0.05] | 0.01 [0.00, 0.08] | 0.00 [0.00, 0.05] |

## Recovery by relation, `all` facts (C_rec with Wilson 95% interval)

C_rec = facts recovered through at least one code route / facts every direct route failed. `chance_expected`: recoveries expected if code answers were guesses from the relation's answer pool (from the cross-subject guess rate). McNemar p per relation, Holm-corrected across relations.

| model | relation | facts | direct_failed | recovered | C_rec_ci | chance_expected | mcnemar_p | p_holm |
|---|---|---|---|---|---|---|---|---|
| cg_oss | atomic_number | 39 | 39 | 8 | 0.20 [0.11, 0.35] | 0.99 | 0.0078 | 0.039 |
| cg_oss | author | 59 | 59 | 0 | 0.00 [0.00, 0.06] | 0.24 | nan | nan |
| cg_oss | capital | 128 | 128 | 2 | 0.02 [0.00, 0.06] | 0.5 | 0.5 | 1.0 |
| cg_oss | currency | 35 | 35 | 1 | 0.03 [0.01, 0.14] | 0.0 | 1.0 | 1.0 |
| cg_oss | founder | 2 | 2 | 0 | 0.00 [0.00, 0.66] | 0.0 | nan | nan |
| cg_oss | inventor | 12 | 12 | 0 | 0.00 [0.00, 0.24] | 0.0 | nan | nan |
| cg_oss | language | 54 | 54 | 16 | 0.30 [0.19, 0.43] | 0.37 | 3.1e-05 | 0.00018 |
| cg_oss | maker | 15 | 15 | 0 | 0.00 [0.00, 0.20] | 0.0 | nan | nan |
| cg_oss | river | 13 | 13 | 0 | 0.00 [0.00, 0.23] | 0.02 | nan | nan |
| cg_oss | sport | 6 | 6 | 0 | 0.00 [0.00, 0.39] | 0.54 | nan | nan |
| cg_oss | symbol | 69 | 69 | 55 | 0.80 [0.69, 0.88] | 0.35 | 5.6e-17 | 3.9e-16 |
| cg_oss | unit | 17 | 17 | 2 | 0.12 [0.03, 0.34] | 0.79 | 0.5 | 1.0 |
| cg_oss | year | 70 | 70 | 1 | 0.01 [0.00, 0.08] | 0.89 | 1.0 | 1.0 |
| cg_oss | ALL | 519 | 519 | 85 | 0.16 [0.13, 0.20] | 4.69 | 5.2e-26 | nan |

## Recovery by relation and route, `all` facts (rate [Wilson 95%] among direct-failed facts)

| model | relation | dict | assert | func | comment |
|---|---|---|---|---|---|
| cg_oss | ALL | 0.09 [0.06, 0.11] | 0.15 [0.12, 0.18] | 0.01 [0.00, 0.02] | 0.00 [0.00, 0.01] |
| cg_oss | atomic_number | 0.05 [0.01, 0.17] | 0.18 [0.09, 0.33] | 0.00 [0.00, 0.09] | 0.00 [0.00, 0.09] |
| cg_oss | author | 0.00 [0.00, 0.06] | 0.00 [0.00, 0.06] | 0.00 [0.00, 0.06] | 0.00 [0.00, 0.06] |
| cg_oss | capital | 0.00 [0.00, 0.03] | 0.02 [0.00, 0.06] | 0.00 [0.00, 0.03] | 0.00 [0.00, 0.03] |
| cg_oss | currency | 0.00 [0.00, 0.10] | 0.00 [0.00, 0.10] | 0.03 [0.01, 0.14] | 0.00 [0.00, 0.10] |
| cg_oss | founder | 0.00 [0.00, 0.66] | 0.00 [0.00, 0.66] | 0.00 [0.00, 0.66] | 0.00 [0.00, 0.66] |
| cg_oss | inventor | 0.00 [0.00, 0.24] | 0.00 [0.00, 0.24] | 0.00 [0.00, 0.24] | 0.00 [0.00, 0.24] |
| cg_oss | language | 0.07 [0.03, 0.18] | 0.28 [0.18, 0.41] | 0.00 [0.00, 0.07] | 0.00 [0.00, 0.07] |
| cg_oss | maker | 0.00 [0.00, 0.20] | 0.00 [0.00, 0.20] | 0.00 [0.00, 0.20] | 0.00 [0.00, 0.20] |
| cg_oss | river | 0.00 [0.00, 0.23] | 0.00 [0.00, 0.23] | 0.00 [0.00, 0.23] | 0.00 [0.00, 0.23] |
| cg_oss | sport | 0.00 [0.00, 0.39] | 0.00 [0.00, 0.39] | 0.00 [0.00, 0.39] | 0.00 [0.00, 0.39] |
| cg_oss | symbol | 0.52 [0.41, 0.64] | 0.78 [0.67, 0.86] | 0.01 [0.00, 0.08] | 0.00 [0.00, 0.05] |
| cg_oss | unit | 0.12 [0.03, 0.34] | 0.00 [0.00, 0.18] | 0.00 [0.00, 0.18] | 0.00 [0.00, 0.18] |
| cg_oss | year | 0.00 [0.00, 0.05] | 0.00 [0.00, 0.05] | 0.01 [0.00, 0.08] | 0.00 [0.00, 0.05] |
