from pytesseract import *
import cv2
import os
import re
import numpy as np
import difflib


def img_similarity(img1, img2):
    data1 = img1.flatten()
    data2 = img2.flatten()
    cnt = 0
    for p1, p2 in zip(data1, data2):
        if p1 == p2:
            cnt += 1
    return cnt / len(data1)


file_list = os.listdir("src/extract/")
file_list.sort()
thumb_list = os.listdir("src/thumbs/")
bound_upper_complete = False
bound_lower_complete = False
height_upper = 920 
height_lower = 1000
pre_word = ""
diff = difflib.Differ()
sample_img = cv2.imread("src/extract/" + file_list[50])
kernel_sharpen = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
add_cnt = 0
page_cnt = 0
thumb_cnt = 0
# selection
print("load preset? [{}:{}] [y/n]".format(str(height_upper), str(height_lower)))
yn = input()
if yn == 'y':
    bound_upper_complete = True
    bound_lower_complete = True
while not bound_upper_complete:
    print("input_upper")
    height_upper = int(input())
    cv2.imshow("height_upper", sample_img[height_upper:, :])
    print("is it ok?[enter/other]")
    ret = cv2.waitKey(0)
    if ret == 13:
        bound_upper_complete = True
    # ok = input()
    # if ok == 'y':
    #     bound_upper_complete = True
    cv2.destroyAllWindows()

while not bound_lower_complete:
    print("input_lower")
    height_lower = int(input())
    cv2.imshow("height_lower", sample_img[height_upper:height_lower, :])
    print("is it ok?[enter/other]")
    ret = cv2.waitKey(0)
    if ret == 13:
        bound_lower_complete = True
    cv2.destroyAllWindows()

result_img = cv2.imread("src/thumbs/" + thumb_list[thumb_cnt])[:height_upper - 5, :]
# result_img = cv2.imread("src/extract/" + file_list[7])[:height_upper - 5, :]

pre_img = cv2.imread("src/extract/" + file_list[0])[height_upper:height_lower, :]
cur_img = cv2.imread("src/extract/" + file_list[0])[height_upper:height_lower, :]
gray = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
inverted = cv2.bitwise_not(gray)
bilateral_filter = cv2.bilateralFilter(inverted, 9, 16, 16)
r, pre_bin = cv2.threshold(bilateral_filter, 127, 255, cv2.THRESH_BINARY)
for file_name in file_list:
    original_img = cv2.imread("src/extract/" + file_name)
    # cv2.imshow("original_img", original_img)

    cur_img = original_img[height_upper:height_lower, :]
    gray = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    bilateral_filter = cv2.bilateralFilter(inverted, 9, 16, 16)
    r, cur_bin = cv2.threshold(inverted, 127, 255, cv2.THRESH_BINARY)
    # cur_bin = cv2.bilateralFilter(cur_bin, 9, 16, 16)
    # dst = cv2.filter2D(bilateral_filter, -1, kernel_sharpen)
    # dst = cur_bin
    dst = bilateral_filter
    new_img = dst
    #
    # gray = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
    # inverted = cv2.bitwise_not(gray)
    # dst = cv2.filter2D(inverted, -1, kernel_sharpen)
    # new_img = dst
    text = image_to_string(dst, lang="Hangul", config="--psm 4 --oem 1")
    # text = image_to_string(dst, lang="kor", config="--psm 4 --oem 1")
    word_list = re.sub("\d+|[ ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎ\{\}\[\]\/?.,;:|\)「＊ㆍ：”…*~`!^\-_+<>@\#$%&\\\=\(\'\"]", "", text).split('\n')
    # word_list = re.sub("\d+|[ \{\}\[\]\/?.,;:|\)*~`!^\-_+<>@\#$%&\\\=\(\'\"]", "", text).split('\n')
    cur_word = max(word_list, key=len)
    img_diff = img_similarity(pre_bin, cur_bin)
    if cur_word != pre_word:
        print(img_diff)
        diff_len = abs(len(list(diff.compare(cur_word, pre_word))) - min(len(cur_word), len(pre_word)))
        if diff_len > len(cur_word) / 4:
            print(cur_word)
            ok = 13
            # if diff_len < 10 or abs(len(cur_word) - len(pre_word)) < 4 or img_diff < 0.05:
            if img_diff > 0.98:
                continue
            if img_diff > 0.9:
                print("Check something")
                cv2.imshow("dst", dst)
                cv2.imshow("cur_bin", cur_bin)
                add_img = cv2.addWeighted(pre_img, 0.5, cur_img, 0.5, 0)
                cv2.imshow("Okay to enter", add_img)
                ok = cv2.waitKey(0)
                cv2.destroyAllWindows()
                # ok = input()
            if ok == 13 and len(cur_word) > 3:
                """ MULTILINE CHECK """
                cur_img2 = original_img[2 * height_upper - height_lower:height_upper, :]
                gray2 = cv2.cvtColor(cur_img2, cv2.COLOR_BGR2GRAY)
                inverted2 = cv2.bitwise_not(gray2)
                bilateral_filter2 = cv2.bilateralFilter(inverted2, 9, 16, 16)
                r, cur_bin2 = cv2.threshold(bilateral_filter2, 127, 255, cv2.THRESH_BINARY)
                # r, cur_bin2 = cv2.threshold(inverted2, 127, 255, cv2.THRESH_BINARY)
                # cur_bin2 = cv2.bilateralFilter(cur_bin2, 9, 16, 16)
                #
                # dst2 = cur_bin2
                dst2 = bilateral_filter2
                text = image_to_string(dst2, lang="Hangul", config="--psm 4 --oem 1")
                # text = image_to_string(dst2, lang="kor", config="--psm 4 --oem 1")
                word_list = re.sub("\d+|[ ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎ\{\}\[\]\/?.,;:|\)「＊ㆍ：”…*~`!^\-_+<>@\#$%&\\\=\(\'\"]", "",
                                   text).split('\n')
                # word_list = re.sub("\d+|[ \{\}\[\]\/?.,;:|\)*~`!^\-_+<>@\#$%&\\\=\(\'\"]", "", text).split('\n')
                cur_word2 = max(word_list, key=len)
                """ MULTILINE CHECK """

                if len(cur_word2) > len(cur_word):
                    print("Check multiline")
                    cv2.imshow("cur_img", cur_img)
                    cv2.imshow("cur_img2", cur_img2)
                    ok = cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    if ok == 13:
                        result_img = np.vstack((result_img, cur_img2))
                    else:
                        result_img = np.vstack((result_img, cur_img))
                else:
                    result_img = np.vstack((result_img, cur_img))

                add_cnt += 1
                if add_cnt % 25 == 0:
                    cv2.imwrite("result-{}.jpg".format(str(page_cnt)), result_img)
                    page_cnt += 1
                    thumb_cnt += 1
                    result_img = cv2.imread("src/thumbs/" + thumb_list[thumb_cnt])[:height_upper - 5, :]
                #
                # if add_cnt % 40 == 0:
                #     cv2.imwrite("result-{}.jpg".format(str(page_cnt)), result_img)
                #     page_cnt += 1
                #     thumb_cnt += 1
                #     result_img = cv2.imread("src/thumbs/" + thumb_list[thumb_cnt])[:height_upper - 5, :]
                # elif add_cnt % 20 == 0:
                #     thumb_cnt += 1
                #     result_img = np.vstack((result_img, cv2.imread("src/thumbs/" + thumb_list[thumb_cnt])[:height_upper - 5, :]))
    pre_word = cur_word
    pre_img = cur_img
    pre_bin = cur_bin
if add_cnt % 25 != 0:
    cv2.imwrite("result-{}.jpg".format(str(page_cnt)), result_img)
