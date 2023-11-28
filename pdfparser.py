from pdfminer.high_level import extract_text
i = 1

# PDF 파일 갯수 만큼 변수 조절
while i <= 150:
    # 넘버링에 맞는 파일이 없어도 실행
    try:
        # PDF 파일이 있는 디렉토리
        text = extract_text("C:/Users\choi4/source/repos/kipriscrawling/kipriscrawling/Threedimensional%d.pdf" % (i))
        # TXT 파일을 저장할 디렉토리
        f = open("C:/Users/choi4/source/repos/Threedimensional%d.txt" % (i), 'w')
        f.write(text)
        f.close()
        i = i+1

    except FileNotFoundError as e:
        print(e)
        i = i+1