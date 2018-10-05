# -*- coding: utf-8 -*-
kResNetAddr= 'http://10.0.0.4:6000'
kInceptionNetAddr= 'http://10.0.0.4:6001'
kBasicConvNetAddr= 'http://10.0.0.4:6002'
kMobileNetAddr= 'http://10.0.0.4:6003'

MODELS = {
    'ResNet50': 0,
    'InceptionV3': 1,
    'BasicConvnet': 2,
	'MobileNet': 3
}

# 임시 수정
# modelIdToAddr= {
# 	MODELS['ResNet50']: kResNetAddr,
# 	MODELS['InceptionV3']: kInceptionNetAddr,
# 	MODELS['BasicConvnet']: kBasicConvNetAddr,
# 	MODELS['MobileNet']: kMobileNetAddr
# }
FILTER_VISUAL_MODE= {"Image":0, "Heatmap":1, "Bar3d":2}

modelIdToAddr= {
	MODELS['ResNet50']: kMobileNetAddr,
	MODELS['InceptionV3']: kInceptionNetAddr,
	MODELS['BasicConvnet']: kBasicConvNetAddr,
	MODELS['MobileNet']: kMobileNetAddr
}