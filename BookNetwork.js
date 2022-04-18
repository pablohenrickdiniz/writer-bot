const tf = require('@tensorflow/tfjs-node-gpu');
const path = require('path');
const fs = require('fs');

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

BookNetwork.cleanText = function(text){
    text = text.trim();
    text = text.split(/[\r\n]+/);
    text = text.map(BookNetwork.cleanLine);
    return text.join("\n");
};

BookNetwork.cleanLine = function(text){
    text = text.trim();
    text = text.split(/\s+/);
    text = text.map(BookNetwork.cleanWord);
    text = text.filter(function(w){
        return w.length > 0;
    });
    text = text.join(' ');
    return text;
};

BookNetwork.cleanWord = function(word){
    return word.trim().split('').map(BookNetwork.cleanCharacter).join('');
};

BookNetwork.cleanCharacter = function(chr){
    return chr.charCodeAt(0) < 32?'':chr.toLowerCase();
};

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
    let textDiv = self.getTextDiv();
    return self.pad(self.textToCharcode(text).map((c) => c/textDiv));
};

BookNetwork.prototype.decodeText = function(encoded){
    let self = this;
    let textDiv = self.getTextDiv();
    let decoded = encoded.map(function(charcode){
        return Math.round(charcode*textDiv);
    }).filter(function(charcode){
        return charcode >= 32;
    });
    return self.charcodeToText(decoded);
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
        let lineDiv = 2;
        let max = Math.max(Object.keys(self.data).map((k) => parseInt(k)).reduce((a,b) => Math.max(a,b)),1);
        while(lineDiv < max){
            lineDiv = Math.pow(lineDiv,2);
        }
        self.lineDiv = lineDiv;
    }
    return self.lineDiv;
};

BookNetwork.prototype.textToCharcode = function(text){
    return text.split('').map((c) => c.charCodeAt(0));
};

BookNetwork.prototype.charcodeToText = function(charcodes){
    return charcodes.map(function(c){
        return String.fromCharCode(c);
    }).join('');
};

BookNetwork.prototype.getTextDiv = function(){
    let self = this;
    if(self.textDiv === null){
        let textDiv = 2;
        let max = Math.max(self.data.map(function(text){
            return self.textToCharcode(text).reduce((a,b) => Math.max(a,b),1);
        }).reduce((a,b) => Math.max(a,b),1),1);
        while(textDiv < max){
            textDiv = Math.pow(textDiv,2);
        }
        self.textDiv = textDiv;
    }
    return  self.textDiv;
};

BookNetwork.prototype.getUnits = function(){
    let self = this;
    if(self.units === null){
        let units = 2;
        let max = self.data.reduce((a,b) => Math.max(a,b.length),2);
        while(units < max){
            units = Math.pow(units,2);
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
                self.encodeLine(line),
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
            let halfUnits = units/2;
            model = tf.sequential({
                layers:[
                    tf.layers.dense({units:halfUnits,inputShape:[1]}),
                    tf.layers.dense({units:halfUnits}),
                    tf.layers.dense({units:units})
                ]
            });
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
    process.exit();
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