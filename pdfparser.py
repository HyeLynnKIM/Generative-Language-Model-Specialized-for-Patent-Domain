from pdfminer.high_level import extract_text
i = 1

# PDF ���� ���� ��ŭ ���� ����
while i <= 150:
    # �ѹ����� �´� ������ ��� ����
    try:
        # PDF ������ �ִ� ���丮
        text = extract_text("C:/Users\choi4/source/repos/kipriscrawling/kipriscrawling/Threedimensional%d.pdf" % (i))
        # TXT ������ ������ ���丮
        f = open("C:/Users/choi4/source/repos/Threedimensional%d.txt" % (i), 'w')
        f.write(text)
        f.close()
        i = i+1

    except FileNotFoundError as e:
        print(e)
        i = i+1