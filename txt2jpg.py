from PIL import ImageDraw, ImageFont, Image 

char_array = ['M','N','H','Q', '$', 'O','C', '?','7','>','!',':','-',';','.',' ']
char_dir = './assets/char_set/'
char_array = ['1','2','3','4','5','6','7','8','9','0','-','_','+','=','~','`','Q',
			'W','E','R','T','Y','U','I','O','P','[',']','{','}','|','A','S','D','F',
			'G','H','J','K','L',';',':','Z','X','C','V','B','N','M','<','>','/','?',
			'!','@','#','$','%','^','&','*','(',')', ' ']
char_dir = "./assets/char_set_full/"
# char_array = ['/','|','\\','-','L','O','#','V','.',' ',':','~','^','!','+','=']
# char_dir = "./assets/char_set_2/"
for i,char in enumerate(char_array):
	canvas = Image.new("RGB", (8,8), (255,255,255))

	draw = ImageDraw.Draw(canvas)
	myfont = ImageFont.truetype("./assets/fonts/RobotoMono-Regular.ttf",8)
	draw.text((2,-1),char, font=myfont, fill=(0,0,0))
	# if i % 5 == 0:
	# 	canvas.show()
	img_title = char_dir + str(i) + ".png"

	canvas.save(img_title, "PNG")

