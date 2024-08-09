# Model Evaluation Metrics

## Model Input
```
Input shape: (128, 128, 3)
```

## Class Labels
```
0: Abra
1: Aerodactyl
2: Alakazam
3: Arbok
4: Arcanine
5: Articuno
6: Beedrill
7: Bellsprout
8: Blastoise
9: Bulbasaur
10: Butterfree
11: Caterpie
12: Chansey
13: Charizard
14: Charmander
15: Charmeleon
16: Clefable
17: Clefairy
18: Cloyster
19: Cubone
20: Dewgong
21: Diglett
22: Ditto
23: Dodrio
24: Doduo
25: Dragonair
26: Dragonite
27: Dratini
28: Drowzee
29: Dugtrio
30: Eevee
31: Ekans
32: Electabuzz
33: Electrode
34: Exeggcute
35: Exeggutor
36: Farfetchd
37: Fearow
38: Flareon
39: Gastly
40: Gengar
41: Geodude
42: Gloom
43: Golbat
44: Goldeen
45: Golduck
46: Graveler
47: Grimer
48: Growlithe
49: Gyarados
50: Haunter
51: Hitmonchan
52: Hitmonlee
53: Horsea
54: Hypno
55: Ivysaur
56: Jigglypuff
57: Jolteon
58: Jynx
59: Kabutops
60: Kadabra
61: Kakuna
62: Kangaskhan
63: Kingler
64: Koffing
65: Lapras
66: Lickitung
67: Machamp
68: Machoke
69: Machop
70: Magikarp
71: Magmar
72: Magnemite
73: Magneton
74: Mankey
75: Marowak
76: Meowth
77: Metapod
78: Mew
79: Mewtwo
80: Moltres
81: Mr. Mime
82: MrMime
83: Nidoking
84: Nidoqueen
85: Nidorina
86: Nidorino
87: Ninetales
88: Oddish
89: Omanyte
90: Omastar
91: Parasect
92: Pidgeot
93: Pidgeotto
94: Pidgey
95: Pikachu
96: Pinsir
97: Poliwag
98: Poliwhirl
99: Poliwrath
100: Ponyta
101: Porygon
102: Primeape
103: Psyduck
104: Raichu
105: Rapidash
106: Raticate
107: Rattata
108: Rhydon
109: Rhyhorn
110: Sandshrew
111: Sandslash
112: Scyther
113: Seadra
114: Seaking
115: Seel
116: Shellder
117: Slowbro
118: Slowpoke
119: Snorlax
120: Spearow
121: Squirtle
122: Starmie
123: Staryu
124: Tangela
125: Tauros
126: Tentacool
127: Tentacruel
128: Vaporeon
129: Venomoth
130: Venonat
131: Venusaur
132: Victreebel
133: Vileplume
134: Voltorb
135: Vulpix
136: Wartortle
137: Weedle
138: Weepinbell
139: Weezing
140: Wigglytuff
141: Zapdos
142: Zubat
```

## Accuracy: 0.7997
## Precision: 0.8010
## Recall: 0.7997
## F1 Score: 0.7987

## Classification Report
```
              precision    recall  f1-score   support

        Abra       0.87      0.67      0.76        49
  Aerodactyl       0.75      0.67      0.71        82
    Alakazam       0.67      0.64      0.65        69
       Arbok       0.94      0.91      0.93       111
    Arcanine       0.74      0.74      0.74        54
    Articuno       0.82      1.00      0.90       145
    Beedrill       0.88      0.87      0.88        69
  Bellsprout       0.90      0.98      0.94       239
   Blastoise       0.82      0.83      0.82       122
   Bulbasaur       0.94      0.92      0.93       102
  Butterfree       0.96      0.98      0.97       183
    Caterpie       0.88      0.91      0.89       109
     Chansey       0.81      0.82      0.82        90
   Charizard       0.79      0.86      0.82        77
  Charmander       0.83      0.77      0.80        95
  Charmeleon       0.78      0.86      0.82       111
    Clefable       0.85      0.79      0.82        81
    Clefairy       0.82      0.77      0.80        84
    Cloyster       0.68      0.75      0.71        67
      Cubone       0.67      0.77      0.72       104
     Dewgong       0.69      0.71      0.70        99
     Diglett       0.79      0.88      0.83        60
       Ditto       0.80      0.80      0.80        64
      Dodrio       0.86      0.67      0.76        76
       Doduo       0.75      0.80      0.77        70
   Dragonair       0.69      0.62      0.65       111
   Dragonite       0.81      0.75      0.78       110
     Dratini       0.61      0.58      0.59       100
     Drowzee       0.86      0.84      0.85        90
     Dugtrio       0.79      0.88      0.83        93
       Eevee       0.72      0.75      0.74       101
       Ekans       0.82      0.90      0.86        99
  Electabuzz       0.65      0.66      0.65        77
   Electrode       0.94      0.98      0.96        51
   Exeggcute       0.86      0.85      0.85       114
   Exeggutor       0.82      0.89      0.86       152
   Farfetchd       0.76      0.84      0.80        96
      Fearow       0.82      0.66      0.73       129
     Flareon       0.79      0.77      0.78       162
      Gastly       0.91      0.89      0.90       133
      Gengar       0.90      0.84      0.87       122
     Geodude       0.78      0.69      0.73        90
       Gloom       0.89      0.83      0.86        65
      Golbat       0.78      0.83      0.80       117
     Goldeen       0.74      0.82      0.78       103
     Golduck       0.74      0.75      0.75       142
    Graveler       0.68      0.75      0.72        72
      Grimer       0.71      0.80      0.75        70
   Growlithe       0.76      0.70      0.73       111
    Gyarados       0.74      0.70      0.72       137
     Haunter       0.82      0.75      0.78        97
  Hitmonchan       0.82      0.73      0.77        88
   Hitmonlee       0.76      0.72      0.74        79
      Horsea       0.76      0.81      0.79        96
       Hypno       0.77      0.80      0.78        70
     Ivysaur       0.93      0.94      0.93       113
  Jigglypuff       0.84      0.84      0.84       153
     Jolteon       0.77      0.71      0.74        93
        Jynx       0.92      0.99      0.95        67
    Kabutops       0.85      0.81      0.83        81
     Kadabra       0.68      0.62      0.65        68
      Kakuna       0.82      0.85      0.84        99
  Kangaskhan       0.88      0.82      0.85        74
     Kingler       0.80      0.69      0.75        95
     Koffing       0.76      0.62      0.68        80
      Lapras       0.83      0.71      0.77       104
   Lickitung       0.67      0.77      0.72        74
     Machamp       0.90      0.74      0.81        76
     Machoke       0.90      0.75      0.82        71
      Machop       0.77      0.78      0.77        77
    Magikarp       0.81      0.81      0.81       105
      Magmar       0.84      0.89      0.86        74
   Magnemite       0.86      0.68      0.76        82
    Magneton       0.91      0.82      0.86        84
      Mankey       0.56      0.57      0.56        67
     Marowak       0.72      0.55      0.62        78
      Meowth       0.97      0.99      0.98       132
     Metapod       0.85      0.89      0.87        85
         Mew       0.86      0.93      0.89       161
      Mewtwo       0.75      0.69      0.72        81
     Moltres       0.75      0.83      0.79        81
    Mr. Mime       0.69      0.73      0.71        33
      MrMime       0.85      0.84      0.84        61
    Nidoking       0.85      0.78      0.82        93
   Nidoqueen       0.80      0.87      0.83        86
    Nidorina       0.66      0.73      0.70        64
    Nidorino       0.74      0.82      0.78        92
   Ninetales       0.73      0.63      0.68       105
      Oddish       0.94      0.96      0.95        99
     Omanyte       0.70      0.75      0.73        83
     Omastar       0.76      0.74      0.75        76
    Parasect       0.90      0.95      0.92        84
     Pidgeot       0.58      0.57      0.58       122
   Pidgeotto       0.57      0.63      0.60       101
      Pidgey       0.69      0.71      0.70       168
     Pikachu       0.77      0.91      0.83       154
      Pinsir       0.76      0.76      0.76        94
     Poliwag       0.86      0.84      0.85       172
   Poliwhirl       0.65      0.72      0.68       126
   Poliwrath       0.70      0.57      0.63        79
      Ponyta       0.65      0.71      0.68        87
     Porygon       0.95      1.00      0.98       146
    Primeape       0.74      0.73      0.73       118
     Psyduck       0.80      0.88      0.84       162
      Raichu       0.80      0.73      0.76       140
    Rapidash       0.70      0.71      0.70       102
    Raticate       0.82      0.73      0.77        98
     Rattata       0.73      0.82      0.77        79
      Rhydon       0.67      0.67      0.67        96
     Rhyhorn       0.75      0.63      0.69        92
   Sandshrew       0.75      0.66      0.70        79
   Sandslash       0.77      0.77      0.77        98
     Scyther       0.82      0.88      0.85       122
      Seadra       0.83      0.87      0.85       158
     Seaking       0.74      0.62      0.67        50
        Seel       0.84      0.65      0.73        40
    Shellder       0.92      0.84      0.88        98
     Slowbro       0.79      0.82      0.81        71
    Slowpoke       0.75      0.75      0.75       107
     Snorlax       0.79      0.82      0.81       136
     Spearow       0.81      0.85      0.83       105
    Squirtle       0.83      0.84      0.83       123
     Starmie       0.92      0.92      0.92       133
      Staryu       0.91      0.83      0.87       112
     Tangela       0.92      0.85      0.88       115
      Tauros       0.80      0.73      0.77       127
   Tentacool       0.81      0.92      0.86        79
  Tentacruel       0.88      0.83      0.85        84
    Vaporeon       0.77      0.80      0.79       115
    Venomoth       0.75      0.64      0.69       120
     Venonat       0.83      0.94      0.88        99
    Venusaur       0.94      0.92      0.93        98
  Victreebel       0.76      0.83      0.79       109
   Vileplume       0.89      0.95      0.92       107
     Voltorb       0.88      0.92      0.90       133
      Vulpix       0.67      0.78      0.72       109
   Wartortle       0.71      0.74      0.73       121
      Weedle       0.78      0.73      0.75       137
  Weepinbell       0.90      0.87      0.88        91
     Weezing       0.70      0.75      0.72       102
  Wigglytuff       0.93      0.85      0.89       131
      Zapdos       0.82      0.84      0.83       109
       Zubat       0.75      0.74      0.74        76

    accuracy                           0.80     14350
   macro avg       0.80      0.79      0.79     14350
weighted avg       0.80      0.80      0.80     14350
```

## Confusion Matrix
![Confusion Matrix](pokemon-cnn-v1.3/confusion_matrix.png)
