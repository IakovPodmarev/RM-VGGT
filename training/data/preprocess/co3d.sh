#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

out_dir="${1:-co3d}"
mkdir -p "$out_dir"
cd "$out_dir"

while IFS=$'\t' read -r file_name cdn_link; do
    if [[ "$file_name" == "file_name" ]]; then
        continue
    fi

    echo "Downloading $file_name"
    curl -L -C - --fail --retry 3 "$cdn_link" -o "$file_name"

    echo "Extracting $file_name"
    unzip -o "$file_name"
    rm -f "$file_name"
done <<'EOF'
file_name	cdn_link
CO3D_apple.zip	https://scontent.xx.fbcdn.net/m1/v/t6/An_QSC6hT-8cb3Gd3PJ9U2VYscdbDWFj_Ny11wi4ptmIspsE70S_BTc8R6OkSBdIZzWNbqbOu6LEyWovGIk.zip?_nc_gid&ccb=10-5&oh=00_AQAqbAttCba2ucrwl3BWV7MCFjCHZZPeIvDHC7GEp-bcuw&oe=6A7424B6&_nc_sid=ba4296
CO3D_backpack.zip	https://scontent.xx.fbcdn.net/m1/v/t6/An9tyyq7fEyndpQjdl4d2UbqMuyGGEYRt32qZRLnxTLLpCQ2PC1QzurnpThigtMS9iS8ggbGL67p7t4aKqg.zip?_nc_gid&ccb=10-5&oh=00_AQCozmmD08GioH0tyLDWk7lt4qoMxKB0bKA3GSzQOxmH3w&oe=6A7406C0&_nc_sid=ba4296
CO3D_ball.zip	https://scontent.xx.fbcdn.net/m1/v/t6/An-jG2tlOIue1umcM2xNXEiGJ89IUpNYdfoa5RTWpBUzaDyS_ZyyUCwEznmuQ6K_cN6THR-IpOCSTXzN6QQ.zip?_nc_gid&ccb=10-5&oh=00_AQCulPY74KI4WYmVMDYMUlB3MgNhQ-KlUQY14w8_l6njyA&oe=6A741646&_nc_sid=ba4296
CO3D_banana.zip	https://scontent.xx.fbcdn.net/m1/v/t6/An9eGc31j9mtSwJJynbI8cCiLecCcwGQ2Q7V5Q9aTcWxHQQqNFbR9LeIL2WmC12AVvzJGPDIHSeRwJVCC1U.zip?_nc_gid&ccb=10-5&oh=00_AQCieNVHCkw8Pak5qEW71vKTC39uVJTlPX1MvSDuHL0whg&oe=6A74085D&_nc_sid=ba4296
CO3D_baseballbat.zip	https://scontent.xx.fbcdn.net/m1/v/t6/An9Hwjr4vTebobXoMz51rKFcl2ucITXLOkYW3_Bj0T5eyiH00u80KE0U2e5WpMFAz49CWFl6qFQKhJ0pt2k.zip?_nc_gid&ccb=10-5&oh=00_AQDiR9BLVggwdo87yQv1odpyd1Aw4bjmSuh0VkWAKgo_wA&oe=6A7407A4&_nc_sid=ba4296
CO3D_baseballglove.zip	https://scontent.xx.fbcdn.net/m1/v/t6/An-RX8sWNUppkI_aRRTS4KzBKNmojiH4yf_Zo_-KpSamp_Br6NiIQrkeEuVQy0qV8qHrfW4NY8GJd0fCXPg.zip?_nc_gid&ccb=10-5&oh=00_AQBUBEmA9NymJ9dw2frVXnP5xpKu8u9AnVAK4LpQyjXjsQ&oe=6A7424FF&_nc_sid=ba4296
CO3D_bench.zip	https://scontent.xx.fbcdn.net/m1/v/t6/An9NIw_tDVj0SM18igW5FfiYlyRVUWGdLK5mh8qguq_JDZ4yqyMElyzRpQw1B_Fto0xnSgzMKcmiNw4IH4w.zip?_nc_gid&ccb=10-5&oh=00_AQAYo1fQ-aSFZ1nx2CbqPt2mLcsT_HaXfCDAX6ztUfh31A&oe=6A73FEB2&_nc_sid=ba4296
CO3D_bicycle.zip	https://scontent.xx.fbcdn.net/m1/v/t6/An_Rm4gpdXvhAhCBYSw6j3LYr8KBiNff_qhJJiIJOBVXpjLki0Ezu5DmB1LITPP3hjlzE5g4qBacdpMNkQw.zip?_nc_gid&ccb=10-5&oh=00_AQAZCa774bmRoG87nkab6up8QS_pVof9YUvqMgKsC4I2cA&oe=6A740E1E&_nc_sid=ba4296
CO3D_book.zip	https://scontent.xx.fbcdn.net/m1/v/t6/An9W1H-8Vz1T584S6E56WuY4M1NAjrnFDC0Qi3ERfIUq0OzDQEv-vYX8t7WX6wP8IIQ_oHeJRYiwFtwTLUM.zip?_nc_gid&ccb=10-5&oh=00_AQBkZVOp7zIqIYX8C2R91ODIQrkahk6EXReHMPjLEu5w-g&oe=6A7409A3&_nc_sid=ba4296
CO3D_bottle.zip	https://scontent.xx.fbcdn.net/m1/v/t6/An_UFwIvyyap4rLtrzCzZB3GFpJb6vd3rocOdyXKxOWVl5AKXAoVbCe5Vs2Z6P_63vYKKt3ji_VsMX6--fc.zip?_nc_gid&ccb=10-5&oh=00_AQAlrr7VK5NmYJ7BxeNIy7ye7RYThI6JoRpYnzxK8y0r9A&oe=6A7425B8&_nc_sid=ba4296
CO3D_bowl.zip	https://scontent.xx.fbcdn.net/m1/v/t6/An9nC5fDvFb2P-r_7k3gCEqrwpNqSL48WBY2_1Zt_10s69DBaB5DaYGLWtqa0nDn-ygxsU1Z_1KLS57sQsY.zip?_nc_gid&ccb=10-5&oh=00_AQDSQSc1yFGvxxC-XPt3eTXVy6fjWLARIx9ggofdBgqH9w&oe=6A7403D5&_nc_sid=ba4296
CO3D_broccoli.zip	https://scontent.xx.fbcdn.net/m1/v/t6/An__h7NUFI_CPHV_vSVVsCQPGvhbAfBnUEQEkNNAi55oHTG5IwQVGCTT5skOAgOF1X_Ez7BGCuW-G2Sn9cQ.zip?_nc_gid&ccb=10-5&oh=00_AQBMU0Qu2Bq4s8q3Jytx-ll_ltcw8zhDg2MBPV0QKcU6PA&oe=6A7415FF&_nc_sid=ba4296
CO3D_cake.zip	https://scontent.xx.fbcdn.net/m1/v/t6/An9GAQcwtTg8srZoI4t1VfIuErH-s_Hhw5_kytqBBTk4hlnat_OH7Ei5ayVGoZVLxO69nR0MLUsKoPhM2fM.zip?_nc_gid&ccb=10-5&oh=00_AQCJKWnNQklf_2L3mUjD3DZTQo6Yo2Om8ahAkocgH0p3aA&oe=6A7420EF&_nc_sid=ba4296
CO3D_car.zip	https://scontent.xx.fbcdn.net/m1/v/t6/An-fxZJgr0X0iIWhTjt-LD_MqBxf0SVGno1ggNYwEkB9zq6nMuQaysGo2nO_T5hvoRX_gDkTEBqSD7GuyGY.zip?_nc_gid&ccb=10-5&oh=00_AQBC1jjj1T46Kco2HUmOLkJbv2CiiDdPAKVZ2pbY3kdY-Q&oe=6A7400AF&_nc_sid=ba4296
CO3D_carrot.zip	https://scontent.xx.fbcdn.net/m1/v/t6/An_U98H5kASgvgePfSCU4dt_uMwydiNuX8AX6mXWZpkEPt85PtzMhNvNzFbuk2L9sszj07GglvTXAtUTHe4.zip?_nc_gid&ccb=10-5&oh=00_AQDzYdJxz2YBV2nMwua_Vu6Sxlsj-t4qfmYeHjy1w48Bug&oe=6A740622&_nc_sid=ba4296
CO3D_cellphone.zip	https://scontent.xx.fbcdn.net/m1/v/t6/An_rwJqQRlwsSV509Iy6uzJuinXA-R78DnrrSl_H4ufdgom5X3E8KLut7xXh-gqOVD-VzomVsOqRbq5A9N0.zip?_nc_gid&ccb=10-5&oh=00_AQCHhCwlf1fy3HEfqkQBTUIHJ_qkXbuikdvfXeUSVUXDOw&oe=6A741B9A&_nc_sid=ba4296
CO3D_chair.zip	https://scontent.xx.fbcdn.net/m1/v/t6/An-kqzDg5pX2UxZbhRA4Hd4d8yWaMzG8C9FauxeYN7jQOz5Tuhg5znQKpVo0VCDnFD7Y1XnnPrZXXySNofc.zip?_nc_gid&ccb=10-5&oh=00_AQAK2_wbw9Xr1-vboeyuGHndPrx9XGq4nwqyKAM9FUat2A&oe=6A742FBE&_nc_sid=ba4296
CO3D_couch.zip	https://scontent.xx.fbcdn.net/m1/v/t6/An-Y531FBt1_Dmjmdml0fAxWVznPEZ4KhVulHN4RqyLIqKI4Fldv1Q2EBkOSRtG8co5l0O9EVtYm894u1-w.zip?_nc_gid&ccb=10-5&oh=00_AQB09HbUZQwklp02yh5Hd77lisCDWLR6MdKLzvzkUUR2sw&oe=6A740FC7&_nc_sid=ba4296
CO3D_cup.zip	https://scontent.xx.fbcdn.net/m1/v/t6/An_IG6IFIimI4F3KOJrIJt8loZF8iELYHDcBrNBxp686y8YTuPeet6hQ_os5K0uI3GnbXQRinE2Y9-304BU.zip?_nc_gid&ccb=10-5&oh=00_AQCU6BnYP6rj4jMaaNIG_gX5CgONxXFNg6dW4eNiEGbiDg&oe=6A73FA49&_nc_sid=ba4296
CO3D_donut.zip	https://scontent.xx.fbcdn.net/m1/v/t6/An8LItMvwmV6Mg5ucjsR8J8ZAW6dNRHzKs3AS6wTX9dhCAPJtPRdxs0E1itjQENEsp404WVyfPJJqW_W2cw.zip?_nc_gid&ccb=10-5&oh=00_AQDBTbPmyArSDx-4ywiViqfoPqlpjL3-OhBuL1nrnBlZUg&oe=6A741581&_nc_sid=ba4296
CO3D_frisbee.zip	https://scontent.xx.fbcdn.net/m1/v/t6/An_0OjAgLVTKieSvSpVQTGPSAb0oqhPRob74uSK2w2lZFSS5i-kTUoTWyc9arl2e4DndFad3qHv8CBDPMFo.zip?_nc_gid&ccb=10-5&oh=00_AQBNyM6zwSVfQkMQ5NjPh7iv_mbg-IDRQOubYgybFejuLg&oe=6A740FB3&_nc_sid=ba4296
CO3D_hairdryer.zip	https://scontent.xx.fbcdn.net/m1/v/t6/An94gPdry5iLgaEbyUt_0EMyFrerhQpSGOPK564nTarWoVwHB2ZVzreuAdEZPniYfnU7sR4NCKVisuiBFpw.zip?_nc_gid&ccb=10-5&oh=00_AQDI8I5P9FC-KOOQwjMVjuD1JgJnXcaYcSthW8g_-Oj9mg&oe=6A742FB7&_nc_sid=ba4296
CO3D_handbag.zip	https://scontent.xx.fbcdn.net/m1/v/t6/An8n_p4JlXoz4XXPX0A3JxxYgpQAs7ALq3bwlQEDERlyaDeebtZUq4TiwasrSTx5atKUnQKAOQv46dG0jsM.zip?_nc_gid&ccb=10-5&oh=00_AQA0vwjoFgqHuuBggWEgL25moIk6Txb6n-JcNihADb1kHg&oe=6A7406F7&_nc_sid=ba4296
CO3D_hotdog.zip	https://scontent.xx.fbcdn.net/m1/v/t6/An9OQMTav9nQ7B9Jhd_H2vq4UF7hvfdHPfVRZltSWqlU-sh9tZgYT_MaWn0-3u9RrgdUnyuKhx_eolT7-W0.zip?_nc_gid&ccb=10-5&oh=00_AQCfrxa7E6xOXEi0lPZ7rfKGzWvVf900hIKOhgaNxS9OkA&oe=6A7422FB&_nc_sid=ba4296
CO3D_hydrant.zip	https://scontent.xx.fbcdn.net/m1/v/t6/An_WQ2hli3BcR5TKb_Gr0A9GsvgVUf8eMwFfWWwKnj2zj1bWAZWHKcp3OfaSZR9gfoYQaEeKV4EpopBLXuk.zip?_nc_gid&ccb=10-5&oh=00_AQDIpOhWZJ7dxtIklYywxRV0PVpgQvVgUHblNd08oVULGA&oe=6A740A45&_nc_sid=ba4296
CO3D_keyboard.zip	https://scontent.xx.fbcdn.net/m1/v/t6/An-oNTCjkPBLtTVYK4qqCun0X6JgPf0su69XKzhbM2Zoks0usg3XY1JD2ukBO-P6uyR0zHrYuVNkd6tS2Ps.zip?_nc_gid&ccb=10-5&oh=00_AQBo2Wnxs-4n4YAxZEpolqPQRraQ3nn-oIDyIaKcP1BBog&oe=6A74145A&_nc_sid=ba4296
CO3D_kite.zip	https://scontent.xx.fbcdn.net/m1/v/t6/An8H26CqbMq5HPuB7C_AIlw6uloZ1N7azpOj6haqJnwgTkGaJBDgFg7siBChwvMVzxjC7oi_UEPFX_2nBqc.zip?_nc_gid&ccb=10-5&oh=00_AQA84U6ifkQuvajR8vs3ssl7bpnlfW-EUfI3yVqMfX0nrg&oe=6A742936&_nc_sid=ba4296
CO3D_laptop.zip	https://scontent.xx.fbcdn.net/m1/v/t6/An9c0OYTwmu5xc9JhqLv5vjnuIRiJnjlG1AS9fo37Smsusw_zuq-fujnYo3M8Tok4DUzhQXV8IRyRYJOh_k.zip?_nc_gid&ccb=10-5&oh=00_AQCXYNO8n4DtvbdeULn1NINV5xLpRt47RRjDCZIMe_Tpyw&oe=6A740347&_nc_sid=ba4296
CO3D_microwave.zip	https://scontent.xx.fbcdn.net/m1/v/t6/An_T1fkClfpPU0DOxdE94ybPsjfI8Nh7KaagsI3IRFojgXlSRt4tBQCHxAcHfEkXuZ2cbvPVZ9RNCbc9B-s.zip?_nc_gid&ccb=10-5&oh=00_AQCXgx_wLNKCkLLJblGWvKM2Z_3T-X2HC2RkXdn4vRMvGw&oe=6A740950&_nc_sid=ba4296
CO3D_motorcycle.zip	https://scontent.xx.fbcdn.net/m1/v/t6/An8OosNfooF0_fEse8t8aN56WkxkD8eAGyKLR3JNHgRNfnTARECZqDsOGlNgpEKvF1vzEY2h1zZLVjtqKrs.zip?_nc_gid&ccb=10-5&oh=00_AQBOtJnY1sNJMzen-Sz7QwinTe1f7BsM15Ea7ngAJ_losA&oe=6A74011D&_nc_sid=ba4296
CO3D_mouse.zip	https://scontent.xx.fbcdn.net/m1/v/t6/An95CAZ2t5EEZaSNq3Cr57NKsYaLCt7y1a_9WHwi0bVobKX9XB4vFGEZLoAO3x5AnrAn6nzG7RsiL1_WFTo.zip?_nc_gid&ccb=10-5&oh=00_AQAMFcwiVNm-VQInNnBTGZYSH0KHxBPaqmO-ILMjpbehzA&oe=6A743014&_nc_sid=ba4296
CO3D_orange.zip	https://scontent.xx.fbcdn.net/m1/v/t6/An8aBfGHNIVQPaaQGHzkzWnEsucc42wu-Ban-VnuMYkylpvQuK-yNA8_EPfN9qDktEDcBz03yb1QFR73Sas.zip?_nc_gid&ccb=10-5&oh=00_AQDNIVS2KEhCCzMJ4dQfCKOwxYsEFS4P0Jf6ZBwTS5s7qw&oe=6A740297&_nc_sid=ba4296
CO3D_parkingmeter.zip	https://scontent.xx.fbcdn.net/m1/v/t6/An_O7b6axtJW94I7Jgb0VFZQo21zwFwZMSA00uxJiCBpTSP2HnU9spp_zQnqZL8cE-FKLLhuWT8MIr5pPcU.zip?_nc_gid&ccb=10-5&oh=00_AQDQRm4Q8qdJ4LlKxEhTGpEU2gPHHfbcUDtjkw7h5BULxA&oe=6A7424A1&_nc_sid=ba4296
CO3D_pizza.zip	https://scontent.xx.fbcdn.net/m1/v/t6/An8CvVz2_o4RPylyeqL3t12hUM9H1zLrk4tK6uv5c3uAO5NIlyNttgVU5iIhwZBL77D-GwHSxcpyOL4NHgE.zip?_nc_gid&ccb=10-5&oh=00_AQCZS3-Q9lNMIVS5QkSe9LXoJDS_cDsdkY19xC8mCD5U4w&oe=6A740427&_nc_sid=ba4296
CO3D_plant.zip	https://scontent.xx.fbcdn.net/m1/v/t6/An8ImEtYSyLU5yCtxIUWI7hejNz51b_NChyCNN5OwFl_V7JCHu41Z1rshq75maZ8nMeaviIK3HooWt0ttEc.zip?_nc_gid&ccb=10-5&oh=00_AQBBPjetMeayYcy4xfQK1KKXuP2pLFZaSUg6MRs-VZtYbQ&oe=6A740EAA&_nc_sid=ba4296
CO3D_remote.zip	https://scontent.xx.fbcdn.net/m1/v/t6/An8qolzrQcDTE_3q2tGM8uEclxVHpqeOmxRCKydENMljfX0PXoT4LSKQ0bfUhS9sGXzl7tzP1MgN9wMIcXU.zip?_nc_gid&ccb=10-5&oh=00_AQD3U0r5zL_48unOkYRclXvI4Ea7AvhTWPa91f22w_X3FQ&oe=6A741915&_nc_sid=ba4296
CO3D_sandwich.zip	https://scontent.xx.fbcdn.net/m1/v/t6/An-dmeRk_G2_7mJ3R-TrZvzf9FjvpProxE0hL1CMAA2jc1GW03eOoqaue__zQDF-pw72DCDfhD-4P72AfK0.zip?_nc_gid&ccb=10-5&oh=00_AQBXHDlw6LS20XIFZ6zf9RbZ0W68-8Vtm6l-GTCdiiZygQ&oe=6A7407A0&_nc_sid=ba4296
CO3D_skateboard.zip	https://scontent.xx.fbcdn.net/m1/v/t6/An8pE5f4i4z6W5Fk1JK_ZtpL2CDEG7siJetk1YDAinUXlY9zNze6Sv4Gj2lLD8noQKQNXs8jUXwROdZAmr4.zip?_nc_gid&ccb=10-5&oh=00_AQDfd6o7RLwqWSAhxKaguyFudeOqagk_DSf2FG_pwSe2ww&oe=6A7416D2&_nc_sid=ba4296
CO3D_stopsign.zip	https://scontent.xx.fbcdn.net/m1/v/t6/An-tR_-d_PK4GDMQBVzhzzazt9ONP4py_TZoy80h4Hea9TPO55fWaL9oHqM72cDfof1zTa-Xkey0sEp5B7M.zip?_nc_gid&ccb=10-5&oh=00_AQDfYzQaUcdW7k80qNx7I9b2lVV2vVWe6EHeyxN6dK9AmQ&oe=6A742B6E&_nc_sid=ba4296
CO3D_suitcase.zip	https://scontent.xx.fbcdn.net/m1/v/t6/An8x0EmZ3Fq_ElbH9O3kno69pwCHHnwCW1nErqgsMAsNv4QQSzV5Naif2hQ6fjiHmXtX7xx1jZdUBLBUwpc.zip?_nc_gid&ccb=10-5&oh=00_AQBydLnXBaZfNJRtrxlisornyS2pcEep5d7HyusklMrokA&oe=6A74301C&_nc_sid=ba4296
CO3D_teddybear.zip	https://scontent.xx.fbcdn.net/m1/v/t6/An9J9NR9EKKW9kr9CH47myJVPREBeLdttjyQYBhSbr3pkRIiox47R9I6swgjg6Yb-6L3AZ4-2LgxVhsjJDc.zip?_nc_gid&ccb=10-5&oh=00_AQC96K7pBbSyjQtdZMBiiabvUtGAYuKtauyRKeNcj0Jw4g&oe=6A7426FB&_nc_sid=ba4296
CO3D_toaster.zip	https://scontent.xx.fbcdn.net/m1/v/t6/An8RczSLJSc71Gg28i5tR4ivaM7MFoKM6fnKe6btpf9tGpMI8IkUEVGRJ12-bmKZh4heFr8MAjSt2WxFtIg.zip?_nc_gid&ccb=10-5&oh=00_AQBha_ZvF-U3C8G7AVhplbCT3L7NWrQnbT7egS7J5xoX7Q&oe=6A742161&_nc_sid=ba4296
CO3D_toilet.zip	https://scontent.xx.fbcdn.net/m1/v/t6/An-BtdPVEYTkaZz_lKz5eQflO84MYFVkrsexLySr0wiw8CBnv_Xcmmnm8hzOsUKMlRqRlbesiPXEEVpKTJE.zip?_nc_gid&ccb=10-5&oh=00_AQD8cmosPunrSh86wLtRwGc-KtWCLxBJPUozhlGtTLHOaw&oe=6A7403B0&_nc_sid=ba4296
CO3D_toybus.zip	https://scontent.xx.fbcdn.net/m1/v/t6/An-e2Mta1zEnPsoQhm2M63T9oADPlecMO8iP8F3s8FBdQDItNZR-djYWoVXDvle7AbVK0pES_xNJlkNcOqo.zip?_nc_gid&ccb=10-5&oh=00_AQC4m2FMy8mKVQvWllPWgYq4joaGjLub8ZcdlPDTIIoH8w&oe=6A74083F&_nc_sid=ba4296
CO3D_toyplane.zip	https://scontent.xx.fbcdn.net/m1/v/t6/An-nluyYsQNpA5H6qo7bDSHX4mtpmie0CCQHnPq7-asIpr6p28VDmHWSekAis6tSNPGaiI2Dx7wl7E6D8CQ.zip?_nc_gid&ccb=10-5&oh=00_AQDEkiAyPGzO_MXLKLyAloKmhdQKBiSAemQ5Yz1aAQX31Q&oe=6A741749&_nc_sid=ba4296
CO3D_toytrain.zip	https://scontent.xx.fbcdn.net/m1/v/t6/An_GnWcAoqhXO6-JvJzgMzsy_ZN1pMZaUlY-K9Yvq95GZDQRZNkXNrco271lErLDAX6IqLexmLNSOjcJAdE.zip?_nc_gid&ccb=10-5&oh=00_AQB9EahfUr9bNF6L9m-fw4lQgVKeVF8XIi0Oe4WZ75sWlQ&oe=6A740C05&_nc_sid=ba4296
CO3D_toytruck.zip	https://scontent.xx.fbcdn.net/m1/v/t6/An_W7VsuDvxZUS0Ms7gMEEhxTJjg6wlrMWIGA2BfNqpTQpSq7VbKAF5eriJQ43_9uG9KqztuYNqyAMwo2FI.zip?_nc_gid&ccb=10-5&oh=00_AQAJSPBhqPfkl8hJJWwOWSikl7KD4Y_i1uarPvuSrlpAwQ&oe=6A742068&_nc_sid=ba4296
CO3D_tv.zip	https://scontent.xx.fbcdn.net/m1/v/t6/An_Dwy0JQM-ff-imtBd_tm1ysnkYvnLVobheO_amBsmzoeU_bmH3l9NQ9F3on4YArNSWXWpccpeIvNJg3iI.zip?_nc_gid&ccb=10-5&oh=00_AQBc5jeRjVCU6I8_mW5y9upTO9oN_rNQN6JBcROI0ovXkg&oe=6A74304A&_nc_sid=ba4296
CO3D_umbrella.zip	https://scontent.xx.fbcdn.net/m1/v/t6/An_fbQlYbVAgMOFjTOY6Evj7IJ2VO6WmCvdX4pqIaHGGkXnJlFDzbg_Il7B_UbrlCwdrYcO2fmIDOnK0jsU.zip?_nc_gid&ccb=10-5&oh=00_AQC6PIwvDWpl9D1PwgF9no0OGHT200Vg0uKzc5SybO7exA&oe=6A7417B2&_nc_sid=ba4296
CO3D_vase.zip	https://scontent.xx.fbcdn.net/m1/v/t6/An9qdS5KidE_caUH0nPX5StCH_u9Xwt2wH3xU6om0p6ZaYK3JAja80iMRG2LHsKCZkm1ul8YoA4KG3LuqWo.zip?_nc_gid&ccb=10-5&oh=00_AQAcRq2WAnKprxN8UIGXXDplWCcXczk23XwvkbkLpyrZCA&oe=6A741DAC&_nc_sid=ba4296
CO3D_wineglass.zip	https://scontent.xx.fbcdn.net/m1/v/t6/An-D7nh5JqEI-3bEtEfyAdCmryr3Zc1mQsd_sFxuIQ6g1E_sYuDerfwJB7j7ZBGa7Wa-I_Uzn3yzaaNPbQ8.zip?_nc_gid&ccb=10-5&oh=00_AQD2nWwhp6DnEdK9vPUKNsYobUBrG0BPZ3U39KanVWaWZw&oe=6A740685&_nc_sid=ba4296
EOF
