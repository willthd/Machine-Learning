# Tips

> Tips for deep learning

## Drop out

* output layer에서는 dropout 반영하지 않는다.

```python
...
x = self.dropout(F.relu(self.fc8(x)))
output = self.fc9(x)
```

* validation 및 test 시에는 dropout 반영되어서는 안되며, model.eval() 이후 dropout 되었던 각 node의 비중 계산이 자동으로 된다.

```python
model.eval()
...
outputs = model(data)
```

