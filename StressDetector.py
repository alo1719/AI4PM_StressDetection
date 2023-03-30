import joblib


class StressDetector:
    def __init__(self):
        data = [721.9018968733334, 727.26728, 74.72231522414693, 12.36126389253763, 12.36106933625897,
                6.044876630233259, 84.12186830059788, 4.933333333333334, 0.0, 8.743512885541394, 105.3109669850863,
                1.262958154031763, -0.7037788098783421, 8.099030355424758e-05, -0.0009511715487198085,
                0.01760474788590069, 0.01120750259732299, 0.01120750077061654, 1.570800250370296, 1.262958154031763,
                -0.7037788098783421, 1016.073759284726, 59.81811700124634, 615.9145731374136, 36.26001524204536,
                90.23971137233677, 66.6170570847812, 3.92186775670829, 9.760288627663208, 1698.605389506921,
                9.245598651311791, 0.1081595727560721, 2.097342416080016, 1.243695691120131]
        model = joblib.load('model.joblib')
        print(model.predict([data])[0])

if __name__ == '__main__':
    detector = StressDetector()