from rule_classifier import classify


blocks = [

{
"text":"Skill 1: Calming the Body and Mind",
"font_name":"Arial-BoldMT",
"font_size":13
},


{
"text":"Introduction",
"font_name":"Arial-BoldMT",
"font_size":12
},


{
"text":"Grounding Practice",
"font_name":"Arial-BoldMT",
"font_size":11
}

]


for block in blocks:
    print(
        block["text"],
        "---->",
        classify(block)
    )