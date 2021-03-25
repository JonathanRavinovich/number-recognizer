import pygame
import numpy as np
import tensorflow as tf


pygame.init()
pygame.font.init()

model = tf.keras.models.load_model('num_detector')


screen = pygame.display.set_mode((800,600))
font = pygame.font.SysFont('Arial', 25)
running = True
hold = False
pixels = []
big = True
eraser = False
number = 0

for i in range(28):
    pixels.append(np.asarray([0.0]*28))

pixels = np.asarray(pixels)
count = 0


while running:

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False
            break

        if event.type == pygame.KEYDOWN:

            if event.key == pygame.K_a and pygame.key.get_mods() & pygame.KMOD_CTRL:
                for x in range(28):
                    for y in range(28):
                        pixels[x][y] = 0

            if event.key == pygame.K_o:
                eraser = not eraser

            if event.key == pygame.K_p:
                big = not big


    if pygame.mouse.get_pressed()[0]:
        hold = True
    else:
        hold = False

    mx,my = pygame.mouse.get_pos()

    if mx<392 and my<392:
        mx = int(mx/14)*14
        my = int(my/14)*14

        if hold:

            if eraser:
                color = 0
            else:
                color = 1


            pixels[int(my/14)][int(mx/14)] = color

            if big:
                if mx/14 != 0:
                    if pixels[int(my/14)][int(mx/14)-1] != 1 and not eraser:
                        pixels[int(my/14)][int(mx/14)-1] = 1
                    elif eraser:
                        pixels[int(my/14)][int(mx/14)-1] = 0

                if mx/14 != 27:
                    if pixels[int(my/14)][int(mx/14)+1] != 1 and not eraser:
                        pixels[int(my/14)][int(mx/14)+1] = 1
                    elif eraser:
                        pixels[int(my/14)][int(mx/14)+1] = 0

                if my/14 != 0:
                    if pixels[int(my/14)-1][int(mx/14)] != 1 and not eraser:
                        pixels[int(my/14)-1][int(mx/14)] = 1
                    elif eraser:
                        pixels[int(my/14)-1][int(mx/14)] = 0

                if my/14 != 27:
                    if pixels[int(my/14)+1][int(mx/14)] != 1 and not eraser:
                        pixels[int(my/14)+1][int(mx/14)] = 1
                    elif eraser:
                        pixels[int(my/14)+1][int(mx/14)] = 0


    screen.fill((180,180,180))

    pygame.draw.rect(screen,(0,0,0),pygame.Rect(0,0,392,392))

    for x in range(28):
        for y in range(28):
            grade = pixels[x][y]
            pygame.draw.rect(screen,(int(255*grade),int(255*grade),int(255*grade)),pygame.rect.Rect(y*14,x*14,14,14))



    if count%100==0:
        predict = model.predict(np.array([np.ravel(pixels)]))
        predict = predict[0]
        number = np.argmax(predict)
        count = 0


    ctrl_surface = font.render('Press ctrl + a to clear the screen',True,(0,0,0))
    ctrl2_surface = font.render('Press p to change the brush size',True,(0,0,0))
    ctrl3_surface = font.render('Press o to toggle the eraser',True,(0,0,0))
    num_surface = font.render('Estimated number: '+str(number),True,(0,0,0))

    if eraser:
        eraser_surface = font.render('Eraser is on',True,(0,0,0))
    else:
        eraser_surface = font.render('Eraser is off',True,(0,0,0))

    if big:
        brush_surface = font.render('Brush size: big',True,(0,0,0))
    else:
        brush_surface = font.render('Brush size: small',True,(0,0,0))

    screen.blit(num_surface, (4,420))
    screen.blit(ctrl_surface, (410,10))
    screen.blit(ctrl2_surface, (410,50))
    screen.blit(ctrl3_surface, (410,90))
    screen.blit(brush_surface, (4,460))
    screen.blit(eraser_surface, (4,500))

    count+=1
    pygame.display.update()


pygame.font.quit()
pygame.quit()

