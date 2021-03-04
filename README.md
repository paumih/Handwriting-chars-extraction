**# Handwriting lines and chars extraction**



handwriting recognition tends to be significantly harder than traditional OCR that uses specific fonts/characters.



The reason this concept is so challenging is that unlike computer fonts, there are nearly infinite variations of handwriting styles. Every one of us has a personal style that is specific and unique.



These variations in handwriting styles pose quite a problem for Optical Character Recognition engines, which are typically trained on computer fonts, not handwriting fonts.



And worse, handwriting recognition is further complicated by the fact that letters can “connect” and “touch” each other, making it incredibly challenging for OCR algorithms to separate them, ultimately leading to incorrect OCR results.



**Lines extraction**
Lines extraction is implemented by the extract_text_lines function which takes as input the resized image (which is 18% of the original size of the image) and saves the extracted lines into the given output directory.


**Characters extraction**
Characters extraction is implemented by the extract_text_chars function which takes as input **the original image** and saves the extracted lines into the given output directory