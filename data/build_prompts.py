"""Builds the Swedish and English prompt sets and splits them into dev/test.

Prompts were written for this study (by Claude, an AI model; see AI_USAGE.md),
grouped by category, and are released under CC0. The split is random with a
fixed seed, made BEFORE any model was run, so the test set cannot have been
chosen to suit any method.
"""
import json
import random
from pathlib import Path

SV = {
    "everyday": [
        "Vad ska jag tänka på när jag köper en begagnad cykel?",
        "Ge mig tre tips för att sova bättre.",
        "Hur kokar man perfekta ägg?",
        "Vad kan jag laga till middag om jag bara har pasta, tomater och lök?",
        "Hur får jag bort en fläck av rödvin från en vit skjorta?",
        "Vilka kläder behöver man för en vandring i fjällen i september?",
        "Hur sparar man pengar på matinköpen?",
        "Ge mig några idéer på vad man kan göra en regnig söndag.",
        "Hur ofta ska man vattna en monstera?",
        "Vad är ett bra sätt att börja springa om man aldrig har tränat?",
        "Hur skriver jag ett bra tackkort till min mormor?",
        "Vad är skillnaden mellan bakpulver och bikarbonat?",
        "Hur förbereder jag mig inför en anställningsintervju?",
        "Vad behöver jag för att börja odla tomater på balkongen?",
        "Hur lär man en hund att sitta?",
        "Vilka är fördelarna med att cykla till jobbet?",
        "Hur planerar man ett barnkalas för tioåringar?",
        "Föreslå en enkel veckomatsedel för en student.",
        "Hur tar jag hand om en gjutjärnspanna?",
        "Vad ska man packa i en första hjälpen-väska?",
    ],
    "facts": [
        "Varför är himlen blå?",
        "Förklara hur fotosyntesen fungerar.",
        "Vad är skillnaden mellan väder och klimat?",
        "Hur fungerar ett vaccin?",
        "Varför har vi årstider?",
        "Vad är en svart hål?",
        "Hur bildas regnbågar?",
        "Vad gör levern i kroppen?",
        "Varför flyter is på vatten?",
        "Hur fungerar demokratin i Sverige?",
        "Vad är allemansrätten?",
        "Berätta kort om Gustav Vasa.",
        "Vad är midsommar och hur firas det?",
        "Varför är Östersjön mindre salt än Atlanten?",
        "Vad är skillnaden mellan en kommun och en region?",
        "Hur fungerar en värmepump?",
        "Vad orsakar norrsken?",
        "Vad är inflation?",
        "Hur fungerar det svenska pensionssystemet i stora drag?",
        "Varför fäller lövträd sina löv på hösten?",
        "Vad är DNA?",
        "Hur uppstod Stockholms skärgård?",
        "Vad är skillnaden mellan ett virus och en bakterie?",
        "Vad innebär offentlighetsprincipen?",
        "Hur fungerar ränta på ett bolån?",
    ],
    "technical": [
        "Förklara vad ett API är.",
        "Vad är skillnaden mellan RAM-minne och hårddisk?",
        "Hur fungerar en neural nätverksmodell, enkelt förklarat?",
        "Vad är molntjänster?",
        "Förklara vad maskininlärning är för en nybörjare.",
        "Hur skapar jag ett starkt lösenord?",
        "Vad är skillnaden mellan Wi-Fi och 5G?",
        "Vad är en databas och varför behövs den?",
        "Förklara vad versionshantering med Git innebär.",
        "Vad är en webbläsares cache?",
        "Hur fungerar kryptering?",
        "Vad är skillnaden mellan Python och JavaScript?",
        "Vad betyder det att en app är öppen källkod?",
        "Hur fungerar en sökmotor?",
        "Vad är en stor språkmodell?",
        "Förklara vad en algoritm är med ett vardagligt exempel.",
        "Vad är phishing och hur skyddar jag mig?",
        "Hur fungerar GPS?",
        "Vad är skillnaden mellan en processor och ett grafikkort?",
        "Vad är blockkedjeteknik?",
        "Förklara vad en VPN gör.",
        "Vad är en Docker-container?",
        "Hur fungerar tvåfaktorsautentisering?",
        "Vad menas med att data är personuppgifter enligt GDPR?",
        "Vad är skillnaden mellan HTTP och HTTPS?",
    ],
    "writing": [
        "Skriv en kort dikt om hösten.",
        "Skriv ett kort mejl där jag sjukanmäler mig från jobbet.",
        "Hitta på en kort saga om en räv och en kråka.",
        "Skriv en kort produktbeskrivning för en termos.",
        "Sammanfatta handlingen i Askungen i tre meningar.",
        "Skriv en inbjudan till en grannträff på lördag.",
        "Skriv en kort motivering till varför man ska läsa böcker.",
        "Formulera en artig påminnelse till en kollega om en försenad rapport.",
        "Skriv tre förslag på namn till ett nytt kafé.",
        "Skriv en kort recension av en påhittad restaurang.",
        "Skriv ett kort tal till en väns 30-årsdag.",
        "Skriv en kort annons för en begagnad soffa.",
        "Skriv början på en deckare som utspelar sig i Göteborg.",
        "Skriv ett kort brev till kommunen om en trasig gatlykta.",
        "Hitta på en gåta och ge svaret.",
        "Skriv en kort LinkedIn-presentation för en nyexaminerad ingenjör.",
        "Skriv en kort text om varför det är viktigt att källsortera.",
        "Skriv en limerick om en katt.",
        "Skriv en kort nyhetsnotis om en ny bibliotek som öppnar.",
        "Skriv en kort instruktion för hur man byter en cykelslang.",
    ],
    "reasoning": [
        "Om ett tåg åker 120 km i timmen, hur långt kommer det på två och en halv timme?",
        "Jag har 3 äpplen och äter ett. Sedan köper jag fem till. Hur många har jag?",
        "Vad är fördelar och nackdelar med att jobba hemifrån?",
        "Jämför att bo i stad och på landet.",
        "Hur skulle du förklara bråktal för ett barn?",
        "Vad är 15 procent av 240?",
        "Om det tar fem maskiner fem minuter att göra fem prylar, hur lång tid tar det för hundra maskiner att göra hundra prylar?",
        "Varför är det viktigt att ha en budget?",
        "Ge argument för och emot sommartid.",
        "Hur kan en liten stad locka fler turister?",
        "Vad borde man tänka på innan man skaffar husdjur?",
        "Lisa är äldre än Per. Per är äldre än Ali. Vem är yngst?",
        "Vilka för- och nackdelar har elbilar jämfört med bensinbilar?",
        "Hur många minuter är det på en vecka?",
        "Varför kan det vara bra att lära sig ett nytt språk som vuxen?",
        "Vad är skillnaden mellan korrelation och kausalitet? Ge ett exempel.",
        "Hur skulle du dela upp en tårta i åtta lika delar med bara tre snitt?",
        "Vad händer med priset om efterfrågan ökar men utbudet är detsamma?",
        "Hur kan man minska sin klimatpåverkan i vardagen?",
        "Vilka frågor bör man ställa innan man skriver på ett hyreskontrakt?",
    ],
}

EN = [
    "What should I think about when buying a used bike?",
    "Give me three tips for sleeping better.",
    "How do you boil the perfect egg?",
    "How do I get a red wine stain out of a white shirt?",
    "How often should I water a monstera?",
    "How do I prepare for a job interview?",
    "Why is the sky blue?",
    "Explain how photosynthesis works.",
    "What is the difference between weather and climate?",
    "How does a vaccine work?",
    "Why do we have seasons?",
    "How do rainbows form?",
    "Why does ice float on water?",
    "What causes the northern lights?",
    "What is inflation?",
    "How does a heat pump work?",
    "Explain what an API is.",
    "What is the difference between RAM and a hard drive?",
    "What are cloud services?",
    "How do I create a strong password?",
    "What is a database and why is it needed?",
    "How does encryption work?",
    "What is a large language model?",
    "What is phishing and how do I protect myself?",
    "How does GPS work?",
    "What does a VPN do?",
    "Write a short poem about autumn.",
    "Write a short email calling in sick to work.",
    "Make up a short fairy tale about a fox and a crow.",
    "Summarize the plot of Cinderella in three sentences.",
    "Write an invitation to a neighbourhood get-together on Saturday.",
    "Suggest three names for a new café.",
    "Write a limerick about a cat.",
    "If a train travels at 120 km per hour, how far does it go in two and a half hours?",
    "What are the pros and cons of working from home?",
    "What is 15 percent of 240?",
    "Lisa is older than Per. Per is older than Ali. Who is the youngest?",
    "How many minutes are there in a week?",
    "What is the difference between correlation and causation?",
    "How can I reduce my carbon footprint in everyday life?",
    "What is the Swedish right of public access (allemansrätten)?",
    "Tell me briefly about Gustav Vasa.",
    "How does democracy work in Sweden?",
    "What is midsummer and how is it celebrated in Sweden?",
    "What is a Docker container?",
    "How does two-factor authentication work?",
    "What is the difference between HTTP and HTTPS?",
    "Explain what version control with Git means.",
    "Write a short product description for a thermos.",
    "Write a short speech for a friend's 30th birthday.",
]


def main(seed: int = 1234):
    rng = random.Random(seed)
    out = Path(__file__).parent
    sv = [{"lang": "sv", "category": c, "prompt": p} for c, ps in SV.items() for p in ps]
    en = [{"lang": "en", "category": "mixed", "prompt": p} for p in EN]
    rng.shuffle(sv)
    rng.shuffle(en)
    n_sv_dev, n_en_dev = 30, 15
    splits = {
        "sv_dev": sv[:n_sv_dev], "sv_test": sv[n_sv_dev:],
        "en_dev": en[:n_en_dev], "en_test": en[n_en_dev:],
    }
    for name, rows in splits.items():
        with open(out / f"prompts_{name}.jsonl", "w", encoding="utf-8") as f:
            for i, r in enumerate(rows):
                f.write(json.dumps({"id": f"{name}_{i:03d}", **r}, ensure_ascii=False) + "\n")
        print(name, len(rows))


if __name__ == "__main__":
    main()
