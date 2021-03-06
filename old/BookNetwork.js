const tf = require('@tensorflow/tfjs-node-gpu');
const path = require('path');
const fs = require('fs');
const TextUtils = require('./TextUtils');

function BookNetwork(name,modelsDir){
    let self = this;
    self.data = [];
    self.trainingData = null;
    self.lineDiv = null;
    self.textDiv = null;
    self.units = null;
    self.model = null;
    self.name = name;
    self.modelsDir = modelsDir;
    self.learningRate = null;
}

BookNetwork.prototype.add = function(line,text){
    let self = this;
    self.data[line] = text;
    self.lineDiv = null;
    self.textDiv = null;
    self.units = null;
    return this;
};

BookNetwork.prototype.remove = function(line){
    let self = this;
    if(self.data[line]){
        delete self.data[line];
        self.lineDiv = null;
        self.textDiv = null;
        self.units = null;
    }
    return this;
};

BookNetwork.prototype.get = function(line){
    let self = this;
    if(self.data[line]){
        return self.data[line];
    }
    return null;
};

BookNetwork.prototype.getEncoded = function(line){
    let self = this;
    if(self.data[line]){
        return self.encodeText(self.data[line]);
    }
    return null;
};

BookNetwork.prototype.encodeText = function(text){
    let self = this;
    return self.pad(TextUtils.encodeText(text,self.getTextDiv()));
};

BookNetwork.prototype.decodeText = function(encoded){
    let self = this;
    return TextUtils.decodeText(encoded,self.getTextDiv());
};

BookNetwork.prototype.pad = function(arr){
    let self = this;
    let units = self.getUnits();
    let tmp = [...arr];
    while(tmp.length < units){
        tmp.push(0);
    }
    return tmp;
};

BookNetwork.prototype.encodeLine = function(line){
    return line/this.getLineDiv();
};

BookNetwork.prototype.decodeLine = function(encoded){
    return Math.round(encoded*this.getLineDiv());
};

BookNetwork.prototype.getLineDiv = function(){
    let self = this;
    if(self.lineDiv === null){
        let keys = Object.keys(self.data);
        for(let i = 0; i < keys.length;i++){
            keys[i] = parseInt(keys[i]);
        }
        max = Math.max(keys.reduce((a,b) => Math.max(a,b)),1);
        keys = null;
        let lineDiv = 0;
        let count = 2;
        while(lineDiv < max){
            lineDiv = Math.pow(2,count);
            count++;
        }
        self.lineDiv = lineDiv;
    }
    return self.lineDiv;
};

BookNetwork.prototype.getTextDiv = function(){
    let self = this;
    if(self.textDiv === null){
        let max = 0;
        self.data.forEach(function(text){
            max = Math.max(TextUtils.charcodesFromText(text).reduce((a,b) => Math.max(a,b),1),max);
        });
        let count = 2;
        let textDiv = 0;
        while(textDiv < max){
            textDiv = Math.pow(2,count);
            count++;
        }
        self.textDiv = textDiv;
    }
    return  self.textDiv;
};

BookNetwork.prototype.getUnits = function(){
    let self = this;
    if(self.units === null){
        let max = self.data.reduce((a,b) => Math.max(a,b.length),2);
        let units = 0;
        let count = 2;
        while(units < max){
            units = Math.pow(2,count);
            count++;
        }
        self.units = units;
    }
    return self.units;
};

BookNetwork.prototype.getID = function(){
    let self = this;
    return [self.getUnits(),self.getLineDiv(),self.getTextDiv()].join('-');
};

BookNetwork.prototype.getTrainigData = function(){
    let self = this;
    if(self.trainingData === null){
        let tmp = [];
        let lines = Object.keys(self.data);
    
        for(let i = 0; i < lines.length;i++){
            let line = lines[i];
            let text = self.data[line];
            tmp.push([
                self.encodeText(text)
            ]);
        }
        self.trainingData = tmp;
    }
    return self.trainingData;
};

BookNetwork.prototype.getModelDir = function(){
    let self = this;
    return path.join(self.modelsDir,self.name,self.getID());
};

BookNetwork.prototype.getModelJsonFile = function(){
    let self = this;
    return path.join(self.modelsDir,self.name,self.getID(),'model.json');
};

BookNetwork.prototype.getModel = async function(){
    let self = this;
    if(self.model === null){
        let model;
        if(fs.existsSync(self.getModelJsonFile())){
            model = await tf.loadLayersModel('file://'+self.getModelJsonFile());
        }
        else{
            let units = self.getUnits();
            model = tf.sequential();
            let rnn = tf.layers.simpleRNN({
                units:1,
                returnSequences:true,
                activation:'linear'
            });
            let inputLayer = tf.input({
                shape:[units,1]
            });
            rnn.apply(inputLayer);
            model.add(rnn);
        }
        model.compile({
            loss:'meanSquaredError',
            optimizer:tf.train.sgd(self.getLearningRate())        
        });
        self.model = model;
    }
    return self.model;
};

BookNetwork.prototype.getLearningRatePath = function(){
    let self = this;
    return path.join(self.getModelDir(),'learningRate.txt');
};

BookNetwork.prototype.saveLearningRate = function(){
    let self = this;
    fs.writeFileSync(self.getLearningRatePath(),String(self.getLearningRate()));
    return self;
};

BookNetwork.prototype.incrementLearningRate = function(){
    let self = this;
    let zeros = 0;
    let tmp = String(self.learningRate).split('.');
    if(tmp[1]){
        for(let i = 0; i < tmp[1].length;i++){
            if(tmp[i] !== "0"){
                break;
            }
            zeros++;
        }
    }
    self.learningRate = 1/Math.pow(10,zeros+1);
    return self;
};

BookNetwork.prototype.getLearningRate = function(){
    let self = this;
    if(self.learningRate === null){
        let learningRate = 0.001;
        if(fs.existsSync(self.getLearningRatePath())){
            learningRate = Number(fs.readFileSync(self.getLearningRatePath(),{encoding:'utf-8'}));
        }
        self.learningRate = learningRate;
    }
    return self.learningRate;
};

BookNetwork.prototype.train = async function(epochs){
    let self = this;
    let trainingData = self.getTrainigData();
    let x = [];
    let y = [];
    for(let i = 0; i < trainingData.length;i++){
        let t = trainingData[i];
        x.push(t[0]);
        y.push(t[1]);
    }
    epochs = epochs || 100;
    let loss = 0;
    do{
        self.model = null;
        let model = await self.getModel();
        let tx = tf.tensor(x);
        let ty = tf.tensor(y);
        await model.fit(tx,ty,{
            epochs:epochs,
            verbose:0,
            callbacks:{
                onEpochEnd:function(logs,b){
                    loss = b.loss;
                    if(logs % 1 === 0){
                        let log = [[
                            logs.toString().padStart(epochs.toString().length,'0'),
                            epochs
                        ].join('/'),'treinando '+self.name+', loss:'+b.loss].join(' - ');
                        if(!isNaN(loss)){
                            console.log(log);
                        }
                    }
                    if(isNaN(loss)){
                        model.stopTraining = true;
                        self.incrementLearningRate();
                    }
                }
            }
        });
    }while(isNaN(loss));
    if(!isNaN(loss)){
        await self.save();
    }
};

BookNetwork.prototype.predict = async function(lines){
    let self = this;
    lines = lines.constructor !== [].constructor?[lines]:lines;
    let model = await self.getModel();
    lines = lines.map((l) => this.encodeLine(l));
    let result = model.predict(tf.tensor(lines,[lines.length,1])).arraySync();
    return result.map(function(encoded){
        return self.decodeText(encoded);
    }).join("\n");
};

BookNetwork.prototype.save = async function(){
    let self = this;
    let model = await self.getModel();
    let modelDir = self.getModelDir();
    if(!fs.existsSync(modelDir)){
        fs.mkdirSync(modelDir,{
            recursive:true
        });
    }
    await model.save('file://'+modelDir);
    self.saveLearningRate();
};

module.exports = BookNetwork;