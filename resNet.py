from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation, Add, MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras import Model

class IdentityBlock(Model):
    def __init__(self, filters, kernal_size):
        super(IdentityBlock, self).__init__(name='')
        self.conv_1 = Conv2D(filters, kernal_size, padding='same')
        self.bn_1 = BatchNormalization()

        self.conv_2 = Conv2D(filters, kernal_size, padding='same')   
        self.bn_1 = BatchNormalization()

        self.act = Activation('relu')
        self.add = Add()
    
    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.bn_1(x)
        x = self.act(x)
        x =  self.conv_2(x)
        x = self.bn_1(x)
        x = self.add([x, inputs])
        return self.act(x)

class ResNet(Model):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        self.conv = Conv2D(64, 7, padding='same')
        self.bn = BatchNormalization()
        self.ac = Activation('relu')
        self.max_pool = MaxPool2D((3, 3))
        self.id1a = IdentityBlock(64,3)
        self.id1b = IdentityBlock(64,3)
        self.global_pool = GlobalAveragePooling2D()
        self.classifier = Dense(num_classes, activation='softmax')
    
    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.ac(x)
        x = self.max_pool(x)

        x = self.id1a(x)
        x = self.id1b(x)
        
        x = self.global_pool(x)
        return self.classifier(x)



resModel = ResNet(10)
# Assuming your input shape is (32, 32, 3)
resModel.build((None, 32, 32, 3))  
resModel.summary()
