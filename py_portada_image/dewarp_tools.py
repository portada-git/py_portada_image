import datetime
from builtins import int
import cv2
import numpy as np
from scipy.optimize import minimize

# Para poder encapsular el código y reaprovechar los procesos y funciones, es preferible usar una clase que se pueda
# configurar de forma independiente y permita definir los cálculos necesarios para las diferentes gestiones de forma
# parcial, evitando así repetir dichos cálculos en cada gestión.

# Los parámetros expresados aquí como constantes deberían formar parte de los atributos de la clase para poderse
# configurar desde una aplicación externa de manera que el valor de los parámetros no dependa de esta librería sino de
# la aplicación inicial que realice las llamadas y puedan ser extraídos, en último término, de un fichero de configuración.
# -------------------------------------------------------------------------------------------------------------------------
# In order to encapsulate the code and reuse the processes and functions, it is preferable to use a class that can be
# configured independently and allows the calculations necessary for the different procedures to be partially defined,
# thus avoiding repeating said calculations in each procedure.
#
# The parameters expressed here as constants should be part of the attributes of the class so that they can be configured
# from an external application so that the value of the parameters does not depend on this library but on the initial
# application that makes the calls and can be extracted, in Lastly, a configuration file.


# Definición de parámetros (TODO: PASAR A ATRIBUTOS DE LA CLASE  DewarpTools siguiendo las recomendaciones de nomenclatura y sintaxis Python)
PAGE_MARGIN_X = 0  # píxeles reducidos para ignorar cerca del borde izquierdo/derecho # OK
PAGE_MARGIN_Y = 0  # píxeles reducidos para ignorar cerca del borde superior/inferior # OK

OUTPUT_ZOOM = 1.0  # cuánto hacer zoom en la salida en relación con la imagen *original*
OUTPUT_DPI = 600  # solo afecta la DPI declarada de la PNG, no la apariencia
REMAP_DECIMATE = 16  # factor de reducción para mapear la imagen

ADAPTIVE_WINSZ = 55  # tamaño de la ventana para umbral adaptativo en píxeles reducidos

TEXT_MIN_WIDTH = 15  # ancho mínimo en píxeles reducidos del contorno de texto detectado
TEXT_MIN_HEIGHT = 2  # altura mínima en píxeles reducidos del contorno de texto detectado
TEXT_MIN_ASPECT = 1.5  # filtrar contornos de texto con una relación w/h por debajo de esto
TEXT_MAX_THICKNESS = 10  # grosor máximo en píxeles reducidos del contorno de texto detectado

EDGE_MAX_OVERLAP = 1.0  # superposición horizontal máxima en píxeles reducidos de contornos en la extensión
EDGE_MAX_LENGTH = 100.0  # longitud máxima en píxeles reducidos de borde que conecta contornos
EDGE_ANGLE_COST = 10.0  # costo de ángulos en bordes (compensación vs. longitud)
EDGE_MAX_ANGLE = 1.0  # cambio máximo de ángulo permitido entre contornos

RVEC_IDX = slice(0, 3)  # índice de rvec en el vector de parámetros
TVEC_IDX = slice(3, 6)  # índice de tvec en el vector de parámetros
CUBIC_IDX = slice(6, 8)  # índice de pendientes cúbicas en el vector de parámetros

SPAN_MIN_WIDTH = 120  # ancho mínimo en píxeles reducidos para la extensión
SPAN_PX_PER_STEP = 80  # espaciado en píxeles reducidos para muestreo a lo largo de las extensiones
FOCAL_LENGTH = 0.4  # longitud focal normalizada de la cámara

DEBUG_LEVEL = 3  # 0=ninguno, 1=algunos, 2=mucho, 3=todo
DEBUG_OUTPUT = 'file'  # archivo, pantalla, ambos

WINDOW_NAME = 'Dewarp'  # Nombre de la ventana para la visualización

# matriz de parámetros intrínsecos por defecto
K = np.array([
    [FOCAL_LENGTH, 0, 0],
    [0, FOCAL_LENGTH, 0],
    [0, 0, 1]], dtype=np.float32)

# Esta función no estaba en el codigo original de Agustín.
def is_cv2image_equal_to(imga, imgb):
    """This function checks if two images are equals"""
    if imga is None or imgb is None:
        ret = imga is None and imgb is None
    else:
        ret = imga.shape == imgb.shape
        if ret:
            difference = cv2.subtract(imga, imgb)
            ret = not np.any(difference)
    return ret

# Las funciones usadas en local mejor ponerlas también dentro de la clase DewarpTools para evitar que puedan usarse
# desde fuera de este módulo. Si fuera posible, sería mejor convertirla en un método privado. Por desgracia Python no
# soporta la privacidad de métodos o funciones, pero se debería seguir la sintaxis y nomenclatura prevista por el lenguaje
# y no documentar.
# ----------------------------------------------------------------------------------------------------------------------------
# It is better to also put the functions used locally inside the DewarpTools class to prevent them from being used from
# outside this module. If possible, it would be better to make it a private method. Unfortunately Python does not support
# the privacy of methods or functions, but the syntax and nomenclature provided by the language should be followed and
# not documented.

# (TODO: PASAR A MÉTODO PRIVADO DE LA CLASE  DewarpTools siguiendo las recomendaciones de nomenclatura y sintaxis Python)
def resize_to_screen(src, maxw=1280, maxh=700, copy=False):
    height, width = src.shape[:2]
    scl_x = float(width) / maxw
    scl_y = float(height) / maxh

    scl = int(np.ceil(max(scl_x, scl_y)))

    if scl > 1.0:
        inv_scl = 1.0 / scl
        img = cv2.resize(src, (0, 0), None, inv_scl, inv_scl, cv2.INTER_AREA)
    elif copy:
        img = src.copy()
    else:
        img = src

    return img


# (TODO: PASAR A MÉTODO PRIVADO DE LA CLASE  DewarpTools siguiendo las recomendaciones de nomenclatura y sintaxis Python)
def get_page_extents(small):
    height, width = small.shape[:2]

    xmin = PAGE_MARGIN_X
    ymin = PAGE_MARGIN_Y
    xmax = width - PAGE_MARGIN_X
    ymax = height - PAGE_MARGIN_Y

    page = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(page, (xmin, ymin), (xmax, ymax), (255, 255, 255), -1)

    outline = np.array([
        [xmin, ymin],
        [xmin, ymax],
        [xmax, ymax],
        [xmax, ymin]])

    return page, outline


# (TODO: PASAR A MÉTODO PRIVADO DE LA CLASE  DewarpTools siguiendo las recomendaciones de nomenclatura y sintaxis Python)
# Para simplificar el código debería eliminarse el parámetro name, pues no sirve para nada. También se debería valorar
# seriamente eliminar el parámetro small i usar un atributo interno de la classe (self._small). Creo que esta función solo se llama
# para procesar la imagen small. Por tanto, es factible eliminarlo como parámetro.
# las funciones de visualización deberían eliminarse pues son muy adecuadas para aprender, pero no son funcionales y en
# producción solo causan ruido.
def get_contours(name, small, pagemask, masktype):
    mask = get_mask(name, small, pagemask, masktype)
    # en algunos entornos/versiones, cv2.findContours aparentemente devuelve una tupla de 2 elementos en lugar de 3
    # https://github.com/facebookresearch/maskrcnn-benchmark/issues/339
    # siempre estamos interesados en el penúltimo elemento de la tupla (el primer o segundo miembro).

    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]

    contours_out = []

    for contour in contours:

        rect = cv2.boundingRect(contour)
        xmin, ymin, width, height = rect

        if (width < TEXT_MIN_WIDTH or
                height < TEXT_MIN_HEIGHT or
                width < TEXT_MIN_ASPECT * height):
            continue

        tight_mask = make_tight_mask(contour, xmin, ymin, width, height)

        if tight_mask.sum(axis=0).max() > TEXT_MAX_THICKNESS:
            continue

        contours_out.append(ContourInfo(contour, rect, tight_mask))

    #    if DEBUG_LEVEL >= 2:
    #        visualize_contours(name, small, contours_out)

    return contours_out


# (TODO: PASAR A MÉTODO PRIVADO DE LA CLASE  DewarpTools siguiendo las recomendaciones de nomenclatura y sintaxis Python)
# Para simplificar el código debería eliminarse el parámetro name, pues no sirve para nada. También se debería valorar
# seriamente eliminar el parámetro small i usar un atributo interno de la classe (self._small). Creo que esta función solo se llama
# para procesar la imagen small. Por tanto, es factible eliminarlo como parámetro.
# Todas las llamadas a los debug_show, son solamente adecuadas para aprender el funcionamiento, pero no son operativas. Deberían eliminarse.
def get_mask(name, small, pagemask, masktype):
    sgray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)

    if masktype == 'text':

        mask = cv2.adaptiveThreshold(sgray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY_INV,
                                     ADAPTIVE_WINSZ,
                                     25)

        #        if DEBUG_LEVEL >= 3:
        #            debug_show(name, 0.1, 'thresholded', mask)

        mask = cv2.dilate(mask, box(9, 1))

        #        if DEBUG_LEVEL >= 3:
        #            debug_show(name, 0.2, 'dilated', mask)

        mask = cv2.erode(mask, box(1, 3))

    #        if DEBUG_LEVEL >= 3:
    #            debug_show(name, 0.3, 'eroded', mask)

    else:

        mask = cv2.adaptiveThreshold(sgray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY_INV,
                                     ADAPTIVE_WINSZ,
                                     7)

        #        if DEBUG_LEVEL >= 3:
        #            debug_show(name, 0.4, 'thresholded', mask)

        mask = cv2.erode(mask, box(3, 1), iterations=3)

        #        if DEBUG_LEVEL >= 3:
        #            debug_show(name, 0.5, 'eroded', mask)

        mask = cv2.dilate(mask, box(8, 2))

    #        if DEBUG_LEVEL >= 3:
    #            debug_show(name, 0.6, 'dilated', mask)

    return np.minimum(mask, pagemask)


# (TODO: PASAR A MÉTODO PRIVADO DE LA CLASE  DewarpTools siguiendo las recomendaciones de nomenclatura y sintaxis Python)
def make_tight_mask(contour, xmin, ymin, width, height):
    tight_mask = np.zeros((height, width), dtype=np.uint8)
    tight_contour = contour - np.array((xmin, ymin)).reshape((-1, 1, 2))

    cv2.drawContours(tight_mask, [tight_contour], 0,
                     (1, 1, 1), -1)

    return tight_mask


# Esta clase está bien, aunque podría encapsularse dentro de la classe DewarpTools. No es indispensable, pero si recomendable
class ContourInfo(object):

    def __init__(self, contour, rect, mask):
        self.contour = contour
        self.rect = rect
        self.mask = mask

        self.center, self.tangent = blob_mean_and_tangent(contour)

        self.angle = np.arctan2(self.tangent[1], self.tangent[0])

        clx = [self.proj_x(point) for point in contour]

        lxmin = min(clx)
        lxmax = max(clx)

        self.local_xrng = (lxmin, lxmax)

        self.point0 = self.center + self.tangent * lxmin
        self.point1 = self.center + self.tangent * lxmax

        self.pred = None
        self.succ = None

    def proj_x(self, point):
        return np.dot(self.tangent, point.flatten() - self.center)

    def local_overlap(self, other):
        xmin = self.proj_x(other.point0)
        xmax = self.proj_x(other.point1)
        return interval_measure_overlap(self.local_xrng, (xmin, xmax))


# (TODO: PASAR A MÉTODO PRIVADO DE LA CLASE  DewarpTools siguiendo las recomendaciones de nomenclatura y sintaxis Python)
def box(width, height):
    return np.ones((height, width), dtype=np.uint8)


# (TODO: PASAR A MÉTODO PRIVADO DE LA CLASE  ContourInfo siguiendo las recomendaciones de nomenclatura y sintaxis Python)
def blob_mean_and_tangent(contour):
    moments = cv2.moments(contour)

    area = moments['m00']

    if area != 0:
        mean_x = moments['m10'] / area
        mean_y = moments['m01'] / area

        moments_matrix = np.array([
            [moments['mu20'], moments['mu11']],
            [moments['mu11'], moments['mu02']]
        ]) / area
    else:
        mean_x = mean_y = 0
        moments_matrix = np.array([
            [0.0, 0.0],
            [0.0, 0.0]
        ])

    _, svd_u, _ = cv2.SVDecomp(moments_matrix)

    center = np.array([mean_x, mean_y])
    tangent = svd_u[:, 0].flatten().copy()

    return center, tangent


# (TODO: PASAR A MÉTODO PRIVADO DE LA CLASE  ContourInfo siguiendo las recomendaciones de nomenclatura y sintaxis Python)
def interval_measure_overlap(int_a, int_b):
    return min(int_a[1], int_b[1]) - max(int_a[0], int_b[0])


# (TODO: PASAR A MÉTODO PRIVADO DE LA CLASE  DewarpTools siguiendo las recomendaciones de nomenclatura y sintaxis Python)
# Para simplificar el código debería eliminarse el parámetro name. Si se decide usar small como atributo de clase debería eliminarse también.
# Como en los casos anteriores deberían elimnarse las visualizaciones.
def assemble_spans(name, small, pagemask, cinfo_list):
    # lista corta
    cinfo_list = sorted(cinfo_list, key=lambda cinfo: cinfo.rect[1])

    # generar todos los bordes candidatos
    candidate_edges = []

    for i, cinfo_i in enumerate(cinfo_list):
        for j in range(i):
            # ten en cuenta que e tiene la forma (puntuación, left_cinfo, right_cinfo)
            edge = generate_candidate_edge(cinfo_i, cinfo_list[j])
            if edge is not None:
                candidate_edges.append(edge)

    # ordenar los bordes candidatos por puntaje (menor es mejor)
    candidate_edges.sort()

    # para cada borde candidato
    for _, cinfo_a, cinfo_b in candidate_edges:
        # si izquierda y derecha no están asignados, unirlos
        if cinfo_a.succ is None and cinfo_b.pred is None:
            cinfo_a.succ = cinfo_b
            cinfo_b.pred = cinfo_a

    # generar lista de extensiones como salida
    spans = []

    # hasta que hayamos eliminado todo de la lista
    while cinfo_list:

        # obtener el primero de la lista
        cinfo = cinfo_list[0]

        # seguir los predecesores hasta que no exista ninguno
        while cinfo.pred:
            cinfo = cinfo.pred

        # comenzar una nueva secuencia
        cur_span = []

        width = 0.0

        # seguir a los sucesores hasta el final de la secuencia
        while cinfo:
            # eliminar de la lista (lamentablemente, esto hace que este bucle también sea O(n^2))
            cinfo_list.remove(cinfo)
            # agregar a la secuencia
            cur_span.append(cinfo)
            width += cinfo.local_xrng[1] - cinfo.local_xrng[0]
            # establecer el sucesor
            cinfo = cinfo.succ

        # agregar si es lo suficientemente largo
        if width > SPAN_MIN_WIDTH:
            spans.append(cur_span)

    #    if DEBUG_LEVEL >= 2:
    #        visualize_spans(name, small, pagemask, spans)

    return spans


# (TODO: PASAR A MÉTODO PRIVADO DE LA CLASE  DewarpTools siguiendo las recomendaciones de nomenclatura y sintaxis Python)
def generate_candidate_edge(cinfo_a, cinfo_b):
    # queremos que 'a' esté a la izquierda de 'b' (por lo que el sucesor de 'a' será 'b'
    # y el predecesor de 'b' será 'a')
    # asegurarse de que el extremo derecho de 'b' esté a la derecha del extremo izquierdo de 'a'.
    if cinfo_a.point0[0] > cinfo_b.point1[0]:
        tmp = cinfo_a
        cinfo_a = cinfo_b
        cinfo_b = tmp

    x_overlap_a = cinfo_a.local_overlap(cinfo_b)
    x_overlap_b = cinfo_b.local_overlap(cinfo_a)

    overall_tangent = cinfo_b.center - cinfo_a.center
    overall_angle = np.arctan2(overall_tangent[1], overall_tangent[0])

    delta_angle = max(angle_dist(cinfo_a.angle, overall_angle),
                      angle_dist(cinfo_b.angle, overall_angle)) * 180 / np.pi

    # queremos que la mayor superposición en x sea pequeña.
    x_overlap = max(x_overlap_a, x_overlap_b)

    dist = np.linalg.norm(cinfo_b.point0 - cinfo_a.point1)

    if (dist > EDGE_MAX_LENGTH or
            x_overlap > EDGE_MAX_OVERLAP or
            delta_angle > EDGE_MAX_ANGLE):
        return None
    else:
        score = dist + delta_angle * EDGE_ANGLE_COST
        return (score, cinfo_a, cinfo_b)


# (TODO: PASAR A MÉTODO PRIVADO DE LA CLASE  DewarpTools siguiendo las recomendaciones de nomenclatura y sintaxis Python)
def angle_dist(angle_b, angle_a):
    diff = angle_b - angle_a

    while diff > np.pi:
        diff -= 2 * np.pi

    while diff < -np.pi:
        diff += 2 * np.pi

    return np.abs(diff)


# (TODO: PASAR A MÉTODO PRIVADO DE LA CLASE  DewarpTools siguiendo las recomendaciones de nomenclatura y sintaxis Python)
def sample_spans(shape, spans):
    span_points = []

    for span in spans:

        contour_points = []

        for cinfo in span:
            yvals = np.arange(cinfo.mask.shape[0]).reshape((-1, 1))
            totals = (yvals * cinfo.mask).sum(axis=0)
            means = totals / cinfo.mask.sum(axis=0)

            xmin, ymin = cinfo.rect[:2]

            step = SPAN_PX_PER_STEP
            start = int(((len(means) - 1) % step) / 2)  # error => convertir a entero (cast float a int)

            contour_points += [(x + xmin, means[x] + ymin)
                               for x in range(start, len(means), step)]

        contour_points = np.array(contour_points,
                                  dtype=np.float32).reshape((-1, 1, 2))

        contour_points = pix2norm(shape, contour_points)

        span_points.append(contour_points)

    return span_points


# (TODO: PASAR A MÉTODO PRIVADO DE LA CLASE  DewarpTools siguiendo las recomendaciones de nomenclatura y sintaxis Python)
def pix2norm(shape, pts):
    height, width = shape[:2]
    scl = 2.0 / (max(height, width))
    offset = np.array([width, height], dtype=pts.dtype).reshape((-1, 1, 2)) * 0.5
    return (pts - offset) * scl


# (TODO: PASAR A MÉTODO PRIVADO DE LA CLASE  DewarpTools siguiendo las recomendaciones de nomenclatura y sintaxis Python)
# Eliminar name i small si se trata como atributo. Eliminar el coóigo de visaulización
def keypoints_from_samples(name, small, pagemask, page_outline,
                           span_points):
    all_evecs = np.array([[0.0, 0.0]])
    all_weights = 0

    for points in span_points:
        _, evec = cv2.PCACompute(points.reshape((-1, 2)),
                                 None, maxComponents=1)

        weight = np.linalg.norm(points[-1] - points[0])

        all_evecs += evec * weight
        all_weights += weight

    evec = all_evecs / all_weights

    x_dir = evec.flatten()

    if x_dir[0] < 0:
        x_dir = -x_dir

    y_dir = np.array([-x_dir[1], x_dir[0]])

    pagecoords = cv2.convexHull(page_outline)
    pagecoords = pix2norm(pagemask.shape, pagecoords.reshape((-1, 1, 2)))
    pagecoords = pagecoords.reshape((-1, 2))

    px_coords = np.dot(pagecoords, x_dir)
    py_coords = np.dot(pagecoords, y_dir)

    px0 = px_coords.min()
    px1 = px_coords.max()

    py0 = py_coords.min()
    py1 = py_coords.max()

    p00 = px0 * x_dir + py0 * y_dir
    p10 = px1 * x_dir + py0 * y_dir
    p11 = px1 * x_dir + py1 * y_dir
    p01 = px0 * x_dir + py1 * y_dir

    corners = np.vstack((p00, p10, p11, p01)).reshape((-1, 1, 2))

    ycoords = []
    xcoords = []

    for points in span_points:
        pts = points.reshape((-1, 2))
        px_coords = np.dot(pts, x_dir)
        py_coords = np.dot(pts, y_dir)
        ycoords.append(py_coords.mean() - py0)
        xcoords.append(px_coords - px0)

    #    if DEBUG_LEVEL >= 2:
    #        visualize_span_points(name, small, span_points, corners)

    return corners, np.array(ycoords), xcoords


# (TODO: PASAR A MÉTODO PRIVADO DE LA CLASE  DewarpTools siguiendo las recomendaciones de nomenclatura y sintaxis Python)
def get_default_params(corners, ycoords, xcoords):
    # ancho y altura de la página
    page_width = np.linalg.norm(corners[1] - corners[0])
    page_height = np.linalg.norm(corners[-1] - corners[0])
    rough_dims = (page_width, page_height)

    # nuestra suposición inicial para la cúbica no tiene pendiente.
    cubic_slopes = [0.0, 0.0]

    # puntos de objeto de una página plana en coordenadas 3D.
    corners_object3d = np.array([
        [0, 0, 0],
        [page_width, 0, 0],
        [page_width, page_height, 0],
        [0, page_height, 0]])

    # estimar rotación y traslación a partir de cuatro correspondencias de puntos 2D a 3D.
    _, rvec, tvec = cv2.solvePnP(corners_object3d,
                                 corners, K, np.zeros(5))

    span_counts = [len(xc) for xc in xcoords]

    params = np.hstack((np.array(rvec).flatten(),
                        np.array(tvec).flatten(),
                        np.array(cubic_slopes).flatten(),
                        ycoords.flatten()) +
                       tuple(xcoords))

    return rough_dims, span_counts, params


# (TODO: PASAR A MÉTODO PRIVADO DE LA CLASE  DewarpTools siguiendo las recomendaciones de nomenclatura y sintaxis Python)
def optimize_params(name, small, dstpoints, span_counts, params):
    keypoint_index = make_keypoint_index(span_counts)

    def objective(pvec):
        ppts = project_keypoints(pvec, keypoint_index)
        return np.sum((dstpoints - ppts) ** 2)

    print('  el objetivo inicial es', objective(params))

    #    if DEBUG_LEVEL >= 1:
    #        projpts = project_keypoints(params, keypoint_index)
    #        display = draw_correspondences(small, dstpoints, projpts)
    #        debug_show(name, 4, 'keypoints before', display)

    print('  optimizando', len(params), 'parámetros...')
    start = datetime.datetime.now()
    res = minimize(objective, params, method='SLSQP')

    end = datetime.datetime.now()
    print('  la optimización tomó', round((end - start).total_seconds(), 2), 'segundos.')
    print('  el objetivo final es', res.fun)
    params = res.x

    #    if DEBUG_LEVEL >= 1:
    #        projpts = project_keypoints(params, keypoint_index)
    #        display = draw_correspondences(small, dstpoints, projpts)
    #        debug_show(name, 5, 'keypoints after', display)

    return params


def project_keypoints(pvec, keypoint_index):
    xy_coords = pvec[keypoint_index]
    xy_coords[0, :] = 0

    return project_xy(xy_coords, pvec)


def project_xy(xy_coords, pvec):
    # obtener los coeficientes del polinomio cúbico dados
    #
    #  f(0) = 0, f'(0) = alpha
    #  f(1) = 0, f'(1) = beta

    alpha, beta = tuple(pvec[CUBIC_IDX])

    poly = np.array([
        alpha + beta,
        -2 * alpha - beta,
        alpha,
        0])

    xy_coords = xy_coords.reshape((-1, 2))
    z_coords = np.polyval(poly, xy_coords[:, 0])

    objpoints = np.hstack((xy_coords, z_coords.reshape((-1, 1))))

    image_points, _ = cv2.projectPoints(objpoints,
                                        pvec[RVEC_IDX],
                                        pvec[TVEC_IDX],
                                        K, np.zeros(5))

    return image_points


def make_keypoint_index(span_counts):
    nspans = len(span_counts)
    npts = sum(span_counts)
    keypoint_index = np.zeros((npts + 1, 2), dtype=int)
    start = 1

    for i, count in enumerate(span_counts):
        end = start + count
        keypoint_index[start:start + end, 1] = 8 + i
        start = end

    keypoint_index[1:, 0] = np.arange(npts) + 8 + nspans

    return keypoint_index


def remap_image(name, img, small, page_dims, params):
    height, width = img.shape[:2]

    print('  la imagen remapeada preservará las dimensiones originales')

    page_x_range = np.linspace(0, page_dims[0], width)
    page_y_range = np.linspace(0, page_dims[1], height)

    page_x_coords, page_y_coords = np.meshgrid(page_x_range, page_y_range)

    page_xy_coords = np.hstack((page_x_coords.flatten().reshape((-1, 1)),
                                page_y_coords.flatten().reshape((-1, 1))))

    page_xy_coords = page_xy_coords.astype(np.float32)

    image_points = project_xy(page_xy_coords, params)
    image_points = norm2pix(img.shape, image_points, False)

    image_x_coords = image_points[:, 0, 0].reshape(page_x_coords.shape)
    image_y_coords = image_points[:, 0, 1].reshape(page_y_coords.shape)

    remapped = cv2.remap(img, image_x_coords, image_y_coords,
                         cv2.INTER_CUBIC,
                         None, cv2.BORDER_REPLICATE)

    # TODO: eliminar la gravación en disco. Solo se debe devolver la imagen
    #    outfile = out_dir + '/' + name + '_mod.jpg'
    #    cv2.imwrite(outfile, remapped)

    #    if DEBUG_LEVEL >= 1:
    #        display = cv2.resize(remapped, (small.shape[1], small.shape[0]),
    #                             interpolation=cv2.INTER_AREA)
    #        debug_show(name, 6, 'output', display)

    #    return outfile
    return remapped


def norm2pix(shape, pts, as_integer):
    height, width = shape[:2]
    scl = max(height, width) * 0.5
    offset = np.array([0.5 * width, 0.5 * height],
                      dtype=pts.dtype).reshape((-1, 1, 2))
    rval = pts * scl + offset
    if as_integer:
        return (rval + 0.5).astype(int)
    else:
        return rval


def get_page_dims(corners, rough_dims, params):
    dst_br = corners[2].flatten()
    dims = np.array(rough_dims)

    def objective(dims):
        proj_br = project_xy(dims, params)
        return np.sum((dst_br - proj_br.flatten())**2)

    res = minimize(objective, dims, method='Powell')
    dims = res.x

#    print('  dimensiones de la página obtenidas', dims[0], 'x', dims[1])

    return dims


class DewarpTools(object):

    def __init__(self, input_path=''):
        if len(input_path) > 0:
            self._image_path = input_path
            self._image = cv2.imread(input_path)
        else:
            self.image = None
            self._image_path = ''
        self._params = None
        self.small = None
        self._corners = None
        self._rough_dims = None
        self._spans = None
        self._are_params_calculated = False

    def __verify_image(self):
        if self.image is None:
            raise Exception("Error: Image is not specified.")

    @property
    def image_path(self):
        return self._image_path

    @image_path.setter
    def image_path(self, val):
        self._image_path = val
        if len(val) > 0:
            self._image = cv2.imread(val)
        else:
            self._image = None

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, val):
        self._image = val
        if val is None:
            self._image_path = ''

    def restart(self, input_path='', image=None):
        if len(input_path) == 0 and image is None:
            self.image_path = input_path
            self.image = image
            self._are_params_calculated = False
        elif image is not None:
            if not is_cv2image_equal_to(image, self.image):
                self.image = image
                self._are_params_calculated = False
        else:
            if input_path != self._image_path:
                self.image_path = input_path
                self._are_params_calculated = False

    def __calculate_params(self):
        self.__verify_image()
        name = 'eliminar_variable_name'
        self._small = resize_to_screen(self.image)

        pagemask, page_outline = get_page_extents(self._small)

        cinfo_list = get_contours(name, self._small, pagemask, 'text')
        self._spans = assemble_spans(name, self._small, pagemask, cinfo_list)

        if len(self._spans) < 3:
            # print('  detectando líneas porque solo hay', len(spans), 'extensiones de texto')
            cinfo_list = get_contours(name, self._small, pagemask, 'line')
            spans2 = assemble_spans(name, self._small, pagemask, cinfo_list)
            if len(spans2) > len(self._spans):
                self._spans = spans2

        # Este trozo de código sobra aquí. Debería usarse para saber si es o no necesario procesar la imagen
        #        if len(spans) < 1:
        #            print('saltando', name, 'porque solo hay', len(spans), 'extensiones')
        #            return None

        span_points = sample_spans(self._small.shape, self._spans)

        self._corners, ycoords, xcoords = keypoints_from_samples(name, self._small,
                                                                 pagemask,
                                                                 page_outline,
                                                                 span_points)

        self._rough_dims, span_counts, params = get_default_params(self._corners,
                                                                   ycoords, xcoords)

        dstpoints = np.vstack((self._corners[0].reshape((1, 1, 2)),) +
                              tuple(span_points))

        self._params = optimize_params(name, self._small,
                                       dstpoints,
                                       span_counts, params)

        self._are_params_calculated = True

    def dewarp_image(self):
        name = 'eliminar_variable_name'
        if not self._are_params_calculated:
            self.__calculate_params()
        if self.is_image_curved():
            page_dims = get_page_dims(self._corners, self._rough_dims, self._params)
            self._image = remap_image(name, self.image, self._small, page_dims, self._params)

    def is_image_curved(self):
        if not self._are_params_calculated:
            self.__calculate_params()
        ret = len(self._spans) >= 1 and abs(self._params[0]) < 0.1 and abs(self._params[1]) < 0.1 and abs(
            self._params[2]) < 0.1
        return ret

    def read_image(self, path):
        """
        read the image from 'path' file
        :param path: the path where the image is
        :return: None
        """
        self.image_path = path

    def save_mage(self, image_path=''):
        """
        Save the image from 'self.image' to 'image_path'. By default, image_path is equal to 'self.image_path'
        :param image_path: the image path where save the image
        :return: None
        """
        self.__verify_image()
        if len(image_path) == 0:
            image_path = self.image_path
        cv2.imwrite(image_path, self.image)


dwp = DewarpTools()


def dewarp_cv2image(cv2img):
    dwp.restart(image=cv2img)
    dwp.dewarp_image()
    return dwp.image


def is_cv2image_curve(cv2img):
    dwp.restart(image=cv2img)
    return dwp.is_image_curved()


def dewarp_cv2image_file(input_path, output_path=''):
    dwp.restart(input_path)
    dwp.dewarp_image()
    dwp.save_mage(output_path)
