import numpy as np

IMAGENET_ALL_CLASSES = [
    'accordion', 'airplane', 'ant', 'antelope', 'apple', 'armadillo',
    'artichoke', 'axe', 'baby_bed', 'backpack', 'bagel', 'balance_beam',
    'banana', 'band_aid', 'banjo', 'baseball', 'basketball', 'bathing_cap',
    'beaker', 'bear', 'bee', 'bell_pepper', 'bench', 'bicycle', 'binder',
    'bird', 'bookshelf', 'bow_tie', 'bow', 'bowl', 'brassiere', 'burrito',
    'bus', 'butterfly', 'camel', 'can_opener', 'car', 'cart', 'cattle',
    'cello', 'centipede', 'chain_saw', 'chair', 'chime', 'cocktail_shaker',
    'coffee_maker', 'computer_keyboard', 'computer_mouse', 'corkscrew',
    'cream', 'croquet_ball', 'crutch', 'cucumber', 'cup_or_mug', 'diaper',
    'digital_clock', 'dishwasher', 'dog', 'domestic_cat', 'dragonfly',
    'drum', 'dumbbell', 'electric_fan', 'elephant', 'face_powder', 'fig',
    'filing_cabinet', 'flower_pot', 'flute', 'fox', 'french_horn', 'frog',
    'frying_pan', 'giant_panda', 'goldfish', 'golf_ball', 'golfcart',
    'guacamole', 'guitar', 'hair_dryer', 'hair_spray', 'hamburger',
    'hammer', 'hamster', 'harmonica', 'harp', 'hat_with_a_wide_brim',
    'head_cabbage', 'helmet', 'hippopotamus', 'horizontal_bar', 'horse',
    'hotdog', 'iPod', 'isopod', 'jellyfish', 'koala_bear', 'ladle',
    'ladybug', 'lamp', 'laptop', 'lemon', 'lion', 'lipstick', 'lizard',
    'lobster', 'maillot', 'maraca', 'microphone', 'microwave', 'milk_can',
    'miniskirt', 'monkey', 'motorcycle', 'mushroom', 'nail', 'neck_brace',
    'oboe', 'orange', 'otter', 'pencil_box', 'pencil_sharpener', 'perfume',
    'person', 'piano', 'pineapple', 'ping-pong_ball', 'pitcher', 'pizza',
    'plastic_bag', 'plate_rack', 'pomegranate', 'popsicle', 'porcupine',
    'power_drill', 'pretzel', 'printer', 'puck', 'punching_bag', 'purse',
    'rabbit', 'racket', 'ray', 'red_panda', 'refrigerator',
    'remote_control', 'rubber_eraser', 'rugby_ball', 'ruler',
    'salt_or_pepper_shaker', 'saxophone', 'scorpion', 'screwdriver',
    'seal', 'sheep', 'ski', 'skunk', 'snail', 'snake', 'snowmobile',
    'snowplow', 'soap_dispenser', 'soccer_ball', 'sofa', 'spatula',
    'squirrel', 'starfish', 'stethoscope', 'stove', 'strainer',
    'strawberry', 'stretcher', 'sunglasses', 'swimming_trunks', 'swine',
    'syringe', 'table', 'tape_player', 'tennis_ball', 'tick', 'tie',
    'tiger', 'toaster', 'traffic_light', 'train', 'trombone', 'trumpet',
    'turtle', 'tv_or_monitor', 'unicycle', 'vacuum', 'violin',
    'volleyball', 'waffle_iron', 'washer', 'water_bottle', 'watercraft',
    'whale', 'wine_bottle', 'zebra'
]

IMAGENET_UNSEEN_CLASSES = ['bench',
    'bow_tie',
    'burrito',
    'can_opener',
    'dishwasher',
    'electric_fan',
    'golf_ball',
    'hamster',
    'harmonica',
    'horizontal_bar',
    'iPod',
    'maraca',
    'pencil_box',
    'pineapple',
    'plate_rack',
    'ray',
    'scorpion',
    'snail',
    'swimming_trunks',
    'syringe',
    'tiger',
    'train',
    'unicycle'
]

VOC_ALL_CLASSES = np.array([
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus','cat',
    'chair', 'cow', 'diningtable', 'horse', 'motorbike','person', 
    'pottedplant', 'sheep', 'tvmonitor','car', 'dog', 'sofa', 'train'
])

##todo
COCO_ALL_CLASSES = np.array([
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
    'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
    'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
    'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush'
])
# COCO_ALL_CLASSES = np.array([
# 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus','cat',
#     'chair', 'cow', 'diningtable', 'horse', 'motorbike','person',
#     'pottedplant', 'sheep', 'tvmonitor','car', 'dog', 'sofa', 'train'
# ])
##

COCO_UNSEEN_CLASSES_65_15 = np.array([
    'airplane', 'train', 'parking_meter', 'cat', 'bear',
    'suitcase', 'frisbee', 'snowboard', 'fork', 'sandwich',
    'hot_dog', 'toilet', 'mouse', 'toaster', 'hair_drier'
])
# COCO_UNSEEN_CLASSES_65_15 = np.array([
# 'car', 'dog', 'sofa', 'train'
# ])

VOC_UNSEEN_CLASSES = np.array(['car', 'dog', 'sofa', 'train'])

COCO_SEEN_CLASSES_48_17 = np.array([
    "toilet",
    "bicycle",
    "apple",
    "train",
    "laptop",
    "carrot",
    "motorcycle",
    "oven",
    "chair",
    "mouse",
    "boat",
    "kite",
    "sheep",
    "horse",
    "sandwich",
    "clock",
    "tv",
    "backpack",
    "toaster",
    "bowl",
    "microwave",
    "bench",
    "book",
    "orange",
    "bird",
    "pizza",
    "fork",
    "frisbee",
    "bear",
    "vase",
    "toothbrush",
    "spoon",
    "giraffe",
    "handbag",
    "broccoli",
    "refrigerator",
    "remote",
    "surfboard",
    "car",
    "bed",
    "banana",
    "donut",
    "skis",
    "person",
    "truck",
    "bottle",
    "suitcase",
    "zebra",
])

COCO_UNSEEN_CLASSES_48_17 = np.array([
    "umbrella",
    "cow",
    "cup",
    "bus",
    "keyboard",
    "skateboard",
    "dog",
    "couch",
    "tie",
    "snowboard",
    "sink",
    "elephant",
    "cake",
    "scissors",
    "airplane",
    "cat",
    "knife"
])


UEC100_ALL_CLASSES = np.array(['rice', 'eels on rice', 'pilaf', "chicken-'n'-egg on rice", 'pork cutlet on rice', 'beef curry', 'sushi', 'chicken rice', 'fried rice', 'tempura bowl', 'bibimbap', 'toast', 'croissant', 'roll bread', 'raisin bread', 'chip butty', 'hamburger', 'pizza', 'sandwiches', 'udon noodle', 'tempura udon', 'soba noodle', 'ramen noodle', 'beef noodle', 'tensin noodle', 'fried noodle', 'spaghetti', 'Japanese-style pancake', 'takoyaki', 'gratin', 'sauteed vegetables', 'croquette', 'grilled eggplant', 'sauteed spinach', 'vegetable tempura', 'miso soup', 'potage', 'sausage', 'oden', 'omelet', 'ganmodoki', 'jiaozi', 'stew', 'teriyaki grilled fish', 'fried fish', 'grilled salmon', 'salmon meuniere ', 'sashimi', 'grilled pacific saury ', 'sukiyaki', 'sweet and sour pork', 'lightly roasted fish', 'steamed egg hotchpotch', 'tempura', 'fried chicken', 'sirloin cutlet ', 'nanbanzuke', 'boiled fish', 'seasoned beef with potatoes', 'hambarg steak', 'beef steak', 'dried fish', 'ginger pork saute', 'spicy chili-flavored tofu', 'yakitori', 'cabbage roll', 'rolled omelet', 'egg sunny-side up', 'fermented soybeans', 'cold tofu', 'egg roll', 'chilled noodle', 'stir-fried beef and peppers', 'simmered pork', 'boiled chicken and vegetables', 'sashimi bowl', 'sushi bowl', 'fish-shaped pancake with bean jam', 'shrimp with chill source', 'roast chicken', 'steamed meat dumpling', 'omelet with fried rice', 'cutlet curry', 'spaghetti meat sauce', 'fried shrimp', 'potato salad', 'green salad', 'macaroni salad', 'Japanese tofu and vegetable chowder', 'pork miso soup', 'chinese soup', 'beef bowl', 'kinpira-style sauteed burdock', 'rice ball', 'pizza toast', 'dipping noodles', 'hot dog', 'french fries', 'mixed rice', 'goya chanpuru'])

UEC100_UNSEEN_CLASSES = np.array(['eels on rice', "chicken-'n'-egg on rice", 'pork cutlet on rice', 'sushi', 'udon noodle', 'beef noodle', 'fried noodle', 'omelet', 'fried fish', 'grilled salmon', 'beef steak', 'chilled noodle', 'roast chicken', 'steamed meat dumpling', 'green salad', 'Japanese tofu and vegetable chowder', 'pork miso soup', 'rice ball', 'pizza toast', 'mixed rice'])


UEC256_ALL_CLASSES = np.array([
'rice', 'eels on rice', 'pilaf', 'chicken-\'n\'-egg on rice', 'pork cutlet on rice', 'beef curry', 'sushi', 'chicken rice', 'fried rice', 'tempura bowl', 'bibimbap', 'toast', 'croissant', 'roll bread', 'raisin bread', 'chip butty', 'hamburger', 'pizza', 'sandwiches', 'udon noodle', 'tempura udon', 'soba noodle', 'ramen noodle', 'beef noodle', 'tensin noodle', 'fried noodle', 'spaghetti', 'Japanese-style pancake', 'takoyaki', 'gratin', 'sauteed vegetables', 'croquette', 'grilled eggplant', 'sauteed spinach', 'vegetable tempura', 'miso soup', 'potage', 'sausage', 'oden', 'omelet', 'ganmodoki', 'jiaozi', 'stew', 'teriyaki grilled fish', 'fried fish', 'grilled salmon', 'salmon meuniere', 'sashimi', 'grilled pacific saury', 'sukiyaki', 'sweet and sour pork', 'lightly roasted fish', 'steamed egg hotchpotch', 'tempura', 'fried chicken', 'sirloin cutlet', 'nanbanzuke', 'boiled fish', 'seasoned beef with potatoes', 'hambarg steak', 'steak', 'dried fish', 'ginger pork saute', 'spicy chili-flavored tofu', 'yakitori', 'cabbage roll', 'omelet2', 'egg sunny-side up', 'natto', 'cold tofu', 'egg roll', 'chilled noodle', 'stir-fried beef and peppers', 'simmered pork', 'boiled chicken and vegetables', 'sashimi bowl', 'sushi bowl', 'fish-shaped pancake with bean jam', 'shrimp with chill source', 'roast chicken', 'steamed meat dumpling', 'omelet with fried rice', 'cutlet curry', 'spaghetti meat sauce', 'fried shrimp', 'potato salad', 'green salad', 'macaroni salad', 'Japanese tofu and vegetable chowder', 'pork miso soup', 'chinese soup', 'beef bowl', 'kinpira-style sauteed burdock', 'rice ball', 'pizza toast', 'dipping noodles', 'hot dog', 'french fries', 'mixed rice', 'goya chanpuru', 'green curry', 'okinawa soba', 'mango pudding', 'almond jelly', 'jjigae', 'dak galbi', 'dry curry', 'kamameshi', 'rice vermicelli', 'paella', 'tanmen', 'kushikatu', 'yellow curry', 'pancake', 'champon', 'crape', 'tiramisu', 'waffle', 'rare cheese cake', 'shortcake', 'chop suey', 'twice cooked pork', 'mushroom risotto', 'samul', 'zoni', 'french toast', 'fine white noodles', 'minestrone', 'pot au feu', 'chicken nugget', 'namero', 'french bread', 'rice gruel', 'broiled eel bowl', 'clear soup', 'yudofu', 'mozuku', 'inarizushi', 'pork loin cutlet', 'pork fillet cutlet', 'chicken cutlet', 'ham cutlet', 'minced meat cutlet', 'thinly sliced raw horsemeat', 'bagel', 'scone', 'tortilla', 'tacos', 'nachos', 'meat loaf', 'scrambled egg', 'rice gratin', 'lasagna', 'Caesar salad', 'oatmeal', 'fried pork dumplings served in soup', 'oshiruko', 'muffin', 'popcorn', 'cream puff', 'doughnut', 'apple pie', 'parfait', 'fried pork in scoop', 'lamb kebabs', 'dish consisting of stir-fried potato, eggplant and green pepper', 'roast duck', 'hot pot', 'pork belly', 'xiao long bao', 'moon cake', 'custard tart', 'beef noodle soup', 'pork cutlet', 'minced pork rice', 'fish ball soup', 'oyster omelette', 'glutinous oil rice', 'trunip pudding', 'stinky tofu', 'lemon fig jelly', 'khao soi', 'Sour prawn soup', 'Thai papaya salad', 'boned, sliced Hainan-style chicken with marinated rice', 'hot and sour, fish and vegetable ragout', 'stir-fried mixed vegetables', 'beef in oyster sauce', 'pork satay', 'spicy chicken salad', 'noodles with fish curry', 'Pork Sticky Noodles', 'Pork with lemon', 'stewed pork leg', 'charcoal-boiled pork neck', 'fried mussel pancakes', 'Deep Fried Chicken Wing', 'Barbecued red pork in sauce with rice', 'Rice with roast duck', 'Rice crispy pork', 'Wonton soup', 'Chicken Rice Curry With Coconut', 'Crispy Noodles', 'Egg Noodle In Chicken Yellow Curry', 'coconut milk soup', 'pho', 'Hue beef rice vermicelli soup', 'Vermicelli noodles with snails', 'Fried spring rolls', 'Steamed rice roll', 'Shrimp patties', 'ball shaped bun with pork', 'Coconut milk-flavored crepes with shrimp and beef', 'Small steamed savory rice pancake', 'Glutinous Rice Balls', 'loco moco', 'haupia', 'malasada', 'laulau', 'spam musubi', 'oxtail soup', 'adobo', 'lumpia', 'brownie', 'churro', 'jambalaya', 'nasi goreng', 'ayam goreng', 'ayam bakar', 'bubur ayam', 'gulai', 'laksa', 'mie ayam', 'mie goreng', 'nasi campur', 'nasi padang', 'nasi uduk', 'babi guling', 'kaya toast', 'bak kut teh', 'curry puff', 'chow mein', 'zha jiang mian', 'kung pao chicken', 'crullers', 'eggplant with garlic sauce', 'three cup chicken', 'bean curd family style', 'salt & pepper fried shrimp with shell', 'baked salmon', 'braised pork meat ball with napa cabbage', 'winter melon soup', 'steamed spareribs', 'chinese pumpkin pie', 'eight treasure rice', 'hot & sour soup'
])

# UEC256_UNSEEN_CLASSES = np.array(['goya chanpuru', 'laulau', 'eels on rice', 'adobo', 'pork loin cutlet', 'meat loaf', 'pork cutlet', 'mango pudding', 'dry curry', 'yellow curry', 'hambarg steak', 'tempura bowl', 'salt & pepper fried shrimp with shell', 'udon noodle', 'sashimi bowl', 'french toast', 'kaya toast', 'potato salad', 'bagel', 'Caesar salad', 'malasada', 'sandwiches', 'french bread', 'pot au feu', 'khao soi', 'tensin noodle', 'fried noodle', 'spaghetti', 'yakitori', 'nasi goreng', 'sauteed spinach', 'seasoned beef with potatoes', 'hot and sour, fish and vegetable ragout', 'stir-fried mixed vegetables', 'miso soup', 'chinese soup', 'minestrone', 'lamb kebabs', 'boned, sliced Hainan-style chicken with marinated rice', 'dried fish', 'cold tofu', 'ginger pork saute', 'simmered pork', 'rice ball', 'moon cake', 'lumpia', 'crullers', 'oshiruko', 'laksa', 'noodles with fish curry', 'Pork Sticky Noodles']
# )

UEC256_UNSEEN_CLASSES = np.array(['eels on rice', 'tempura bowl', 'sandwiches', 'udon noodle', 'tensin noodle', 'fried noodle', 'spaghetti', 'sauteed spinach', 'miso soup', 'seasoned beef with potatoes', 'hambarg steak', 'dried fish', 'ginger pork saute', 'yakitori', 'cold tofu', 'simmered pork', 'sashimi bowl', 'potato salad', 'chinese soup', 'rice ball', 'goya chanpuru', 'mango pudding', 'dry curry', 'yellow curry', 'french toast', 'minestrone', 'pot au feu', 'french bread', 'pork loin cutlet', 'bagel', 'meat loaf', 'Caesar salad', 'oshiruko', 'lamb kebabs', 'moon cake', 'pork cutlet', 'khao soi', 'boned, sliced Hainan-style chicken with marinated rice', 'hot and sour, fish and vegetable ragout', 'stir-fried mixed vegetables', 'noodles with fish curry', 'Pork Sticky Noodles', 'malasada', 'laulau', 'adobo', 'lumpia', 'nasi goreng', 'laksa', 'kaya toast', 'crullers', 'salt & pepper fried shrimp with shell']
)


def get_class_labels(dataset):
    if dataset == 'coco':
        return COCO_ALL_CLASSES
    elif dataset == 'voc':
        return VOC_ALL_CLASSES
    elif dataset == 'imagenet':
        return IMAGENET_ALL_CLASSES
    elif dataset == 'uec256':
        return UEC256_ALL_CLASSES
    elif dataset == 'uec100':
        return UEC100_ALL_CLASSES

##todo
def get_unseen_class_labels(dataset, split='65_15'):
    if dataset == 'coco':
        # return COCO_UNSEEN_CLASSES_65_15 if split=='65_15' else COCO_UNSEEN_CLASSES_48_17
        return COCO_UNSEEN_CLASSES_65_15
    elif dataset == 'voc':
        return VOC_UNSEEN_CLASSES
    elif dataset == 'imagenet':
        return IMAGENET_UNSEEN_CLASSES
    elif dataset == 'uec256':
        return UEC256_UNSEEN_CLASSES
    elif dataset == 'uec100':
        return UEC100_UNSEEN_CLASSES

##
def get_unseen_class_ids(dataset, split='65_15'):
    if dataset == 'coco':
        return get_unseen_coco_cat_ids(split)
    elif dataset == 'voc':
        return get_unseen_voc_ids()
    elif dataset == 'imagenet':
        return get_unseen_imagenet_ids()
    elif dataset == 'uec256':
        return get_unseen_uec256_ids()
    elif dataset == 'uec100':
        return get_unseen_uec100_ids()

def get_seen_class_ids(dataset, split='65_15'):
    if dataset == 'coco':
        return get_seen_coco_cat_ids(split)
    elif dataset == 'voc':
        return get_seen_voc_ids()
    elif dataset == 'imagenet':
        return get_seen_imagenet_ids()
    elif dataset == 'uec256':
        return get_seen_uec256_ids()
    elif dataset == 'uec100':
        return get_seen_uec100_ids()

def get_unseen_uec256_ids(split='205_51'):
    UNSEEN_CLASSES = UEC256_UNSEEN_CLASSES
    ids = np.where(np.isin(UEC256_ALL_CLASSES, UNSEEN_CLASSES))[0] + 1
    return ids

def get_seen_uec256_ids(split='205_51'):
    seen_classes = np.setdiff1d(UEC256_ALL_CLASSES, UEC256_UNSEEN_CLASSES)
    ids = np.where(np.isin(UEC256_ALL_CLASSES, seen_classes))[0] + 1
    return ids

def get_unseen_uec100_ids(split='80_20'):
    UNSEEN_CLASSES = UEC100_UNSEEN_CLASSES
    ids = np.where(np.isin(UEC100_ALL_CLASSES, UNSEEN_CLASSES))[0] + 1
    return ids

def get_seen_uec100_ids(split='80_20'):
    seen_classes = np.setdiff1d(UEC100_ALL_CLASSES, UEC100_UNSEEN_CLASSES)
    ids = np.where(np.isin(UEC100_ALL_CLASSES, seen_classes))[0] + 1
    return ids

##todo
# def get_unseen_coco_cat_ids(split='65_15'):
#     UNSEEN_CLASSES = COCO_UNSEEN_CLASSES_65_15 if split=='65_15' else COCO_UNSEEN_CLASSES_48_17
#     ids = np.where(np.isin(COCO_ALL_CLASSES, UNSEEN_CLASSES))[0] + 1
#     return ids
def get_unseen_coco_cat_ids(split='65_15'):
    UNSEEN_CLASSES = COCO_UNSEEN_CLASSES_65_15
    ids = np.where(np.isin(COCO_ALL_CLASSES, UNSEEN_CLASSES))[0] + 1
    return ids
##
##todo
# def get_seen_coco_cat_ids(split='65_15'):
#
#     seen_classes = np.setdiff1d(COCO_ALL_CLASSES, COCO_UNSEEN_CLASSES_65_15) if split=='65_15' else COCO_SEEN_CLASSES_48_17
#     ids = np.where(np.isin(COCO_ALL_CLASSES, seen_classes))[0] + 1
#     return ids
def get_seen_coco_cat_ids(split='65_15'):

    seen_classes = np.setdiff1d(COCO_ALL_CLASSES, COCO_UNSEEN_CLASSES_65_15)
    ids = np.where(np.isin(COCO_ALL_CLASSES, seen_classes))[0] + 1
    return ids
##

def get_unseen_voc_ids():
    ids = np.where(np.isin(VOC_ALL_CLASSES, VOC_UNSEEN_CLASSES))[0] + 1
    # ids = np.concatenate(([0], ids+1))
    return ids

def get_seen_voc_ids():
    seen_classes = np.setdiff1d(VOC_ALL_CLASSES, VOC_UNSEEN_CLASSES)
    ids = np.where(np.isin(VOC_ALL_CLASSES, seen_classes))[0] + 1
    # ids = np.concatenate(([0], ids+1))
    return ids

def get_unseen_imagenet_ids():
    ids = np.where(np.isin(IMAGENET_ALL_CLASSES, IMAGENET_UNSEEN_CLASSES))[0] + 1
    return ids

def get_seen_imagenet_ids():
    seen_classes = np.setdiff1d(IMAGENET_ALL_CLASSES, IMAGENET_UNSEEN_CLASSES)
    ids = np.where(np.isin(IMAGENET_ALL_CLASSES, seen_classes))[0] + 1
    return ids
