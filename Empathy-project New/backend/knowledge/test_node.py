from node_builder import build_node


block = {

    "page":1,

    "text":"Skill 1: Calming the Body and Mind",

    "font_name":"Arial-BoldMT",

    "font_size":13

}


node = build_node(
    block,
    "chapter",
    "node_001"
)


print(node)