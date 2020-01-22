class AddLayer():
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y

        return out

    def backward(self, dout):
        dx = dout
        dy = dout

        return dx, dy

apple = 100
orage = 500
banana = 300


# layer
add_layer1 = AddLayer()
add_layer2 = AddLayer()


# forward
apple_orange = add_layer1.forward(apple, orage)
print(apple_orange)
total = add_layer2.forward(apple_orange, banana)
print(total)

# backward
dprice = 1
apple_orange, banana = add_layer2.backward(dprice)
print(apple_orange, banana)
apple, orange = add_layer1.backward(apple_orange)
print(apple, orange)
