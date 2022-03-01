from docxtpl import DocxTemplate

lista=['Język Obcy', 'Matematyka', 'Fizyka','Chemia']

for plik in lista:
    template=DocxTemplate(r'sylabus_1_st.docx')
    context={
        'rok': '2021/2022',
        'nazwa':plik,
        'sm':1
    }
    template.render(context)
    template.save(rf'D:\PROGRAMY\PYTHON\CENTRUM EDUKACJI IIS\GDYNIA\NN\ML_Gdynia\automatyzacja\target\{plik}.docx')