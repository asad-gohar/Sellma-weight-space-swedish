# Spot check of judge grades (by the experimenter, before arm names were applied)

10 Sheet-A items drawn with `random.Random(3).sample(sorted(item_ids), 10)`. For each, the experimenter (Claude, the main agent) read the prompt and answer and compared them with the judge's grades. It agreed with all 10.

| item | prompt | judge fluency | judge adequacy | agree |
|---|---|---|---|---|
| A060 | Hur fungerar demokratin i Sverige? | 1 | 0 | yes |
| A151 | Skriv en kort instruktion för hur man byter en cykelslang. | 1 | 0 | yes |
| A139 | Vad är inflation? | 1 | 0 | yes |
| A033 | Hur fungerar en sökmotor? | 1 | 0 | yes |
| A094 | Förklara hur fotosyntesen fungerar. | 1 | 0 | yes |
| A234 | Hur skulle du förklara bråktal för ett barn? | 0 | 0 | yes |
| A154 | Vad är en svart hål? | 1 | 0 | yes |
| A121 | Hur fungerar en sökmotor? | 1 | 0 | yes |
| A160 | Förklara vad maskininlärning är för en nybörjare. | 1 | 0 | yes |
| A148 | Skriv en kort instruktion för hur man byter en cykelslang. | 1 | 0 | yes |
