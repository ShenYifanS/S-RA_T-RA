#!/bin/bash
CSV_FILE="./example_dataset/file.csv"
tail -n +2 "$CSV_FILE" | while IFS=, read -r filename minx miny maxx maxy inputx inputy
do
    box="$(printf "%.0f,%.0f,%.0f,%.0f" "$minx" "$maxx" "$miny" "$maxy")"
    int_inputx=${inputx%.*}
    int_inputy=${inputy%.*}

    # sam_vit_l_0b3195.pth, sam_vit_b_01ec64.pth, sam_vit_h_4b8939.pth
    python SSAscripts/attack.py \
        --orig_image "example_dataset/$filename" \
        --sam_model vit_b \
        --checkpoint checkpoints/sam_vit_b_01ec64.pth \
        --box "$box" \
        --epsilon 8 \
        --apply_ssa \
        --sam_checkpoint checkpoints/sam_vit_h_4b8939.pth \
        --input_point $int_inputx $int_inputy \
        --result_csv SSAscripts/B_H_True_0.1_8_50.csv \
        --rho 0.1 \
        --mi -1
done
