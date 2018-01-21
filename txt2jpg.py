from PIL import ImageDraw, ImageFont, Image 

# chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890~!@#$%^&*()-_=+[]{}\\|<>'
char_array = ['M','N','H','Q', '$', 'O','C', '?','7','>','!',':','-',';','.',' ']
char_dir = "./assets/char_set/"
for i,char in enumerate(char_array):
	canvas = Image.new("RGB", (8,8), (255,255,255))

	draw = ImageDraw.Draw(canvas)

	myfont = ImageFont.truetype("./assets/fonts/RobotoMono-Regular.ttf",8)
	draw.text((2,-1),char, font=myfont, fill=(0,0,0))

	if i % 5 == 0:
		canvas.show()

	img_title = char_dir + str(i) + ".jpg"

	canvas.save(img_title, "JPEG")

