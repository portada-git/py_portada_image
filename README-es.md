# py_portada_image

([Leer en inglés](README.md))

Las imagenes de texto impreso puden presentar anomalías como transparencias del reveso, líneas inclinadas o curvadas, carácteres poco o excesivamente marcados, etc. Habitualmente, estos defectos en la calidad de la imagenes suelen interferir negativamente en el proceso de reconociemento de texto (OCR). Aquí presentamos un conjunto de herramientas para comprobar y corregir algunas estas irregularidades a fin de mejorar su calidad y minimizar así, la tasa de errores durante el proceso de reconociemento de texto.

## Módulo deskew_tools 

Este módulo dispone de dos funciones bàsicas para comprobar y corregir la inclinación de las líneas de texto en una imagen a fin de conseguir texto alineado horizontalmente. Las funciones son:

- ___isSkimageSkewed(skimage, min_angle=0)___. Esta función evalua si la imagen pasada como paràmetro (skimage) corresponde a un documento de líneas de texto, las cuales presentan una inclinación mayor que el valor ángula pasado mediante el parámetro llamado _min_angle_. 
- ___deskewSkimage(skimage)___. La función _deskewSkimage_ corrige la inclinación de las líneas de texto de la imagen pasada como parámetro a fin de conseguir que que estas se visualicen horizontales (sin inclinación). 

Ambas funciones precisan recibir una imagen compatible con la cargada por skimage.io 

