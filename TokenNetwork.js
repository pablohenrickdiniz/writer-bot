const tf = require('@tensorflow/tfjs-node-gpu');
const path = require('path');
const fs = require('fs');
const TextUtils = require('./TextUtils');

function TokenNetwork(options){
    let self = this;
    initialize(self);
    options = options || {};
    Object.keys(options).forEach(function(k){
        self[k] = options[k];
    });
}

TokenNetwork.prototype.loadTextFile = function(filename){
    let self = this;
    self.loadText(fs.readFileSync(filename,{encoding:'utf-8'}));
    return self;
};

TokenNetwork.prototype.loadText = function(text){
    let self = this;
    text = TextUtils.cleanText(text);
    self.characters = self.characters.concat(TextUtils.charactersFromText(text));
    self.data = TextUtils.charcodesFromText(text);
    self.tokens = null;
    return this;
};

TokenNetwork.prototype.lineTo3d = function(l){
    let self = this;
    let line = [...l];
    return tf.tensor(line,[1,line.length,1]).arraySync()[0];
};


TokenNetwork.prototype.train = async function(epochs,step){
    let self = this;
    let trainingData = self.trainingData;
    let x = [];
    let y = [];

    for(let i = 0; i < trainingData.length && i < 10; i++){
        x.push(trainingData[i][0]);
        y.push(trainingData[i][1]);
    }
   
    epochs = epochs || 100;
    let loss = 0;
    do{
        let model = self.model;
        let tx = tf.tensor(x);
        let ty = tf.tensor(y);

        await model.fit(tx,ty,{
            epochs:epochs,
          //  batchSize: 32,
            verbose:0,
            callbacks:{
                onEpochEnd:function(logs,b){
                    loss = b.loss;
                    if(!isNaN(loss) && step){
                        self.loss = loss;
                        step(logs+1,epochs,loss);
                    }
                    else{
                        model.stopTraining = true;
                        self.incrementLearningRate();
                        self.model = null;
                    }
                }
            },
            
        });
    }
    while(isNaN(loss));
};

TokenNetwork.prototype.predict = function(text){
    let self = this;
    let batchSize = self.batchSize;
    text = text.substring(0,batchSize);
    text = text.padEnd(batchSize, ' ');
    let input = TextUtils.charcodesFromText(text);
    let model = self.model;
    let predict = model.predict( tf.tensor([self.lineTo3d(input)])).flatten().round().arraySync();
    return TextUtils.textFromCharcodes(predict);
};

TokenNetwork.prototype.toJSON = function(){
    let self = this;
    return {
        learningRate:self.learningRate,
        loss:self.loss,
        characters:self.characters,
        letterSeparators:self.letterSeparators,
        paragraphSeparators:self.paragraphSeparators,
        tokenSeparators:self.tokenSeparators
    };
};

TokenNetwork.prototype.save = async function(outputdir){
    let self = this;
    let model = self.model;
    if(!fs.existsSync(outputdir)){
        fs.mkdirSync(outputdir,{
            recursive:true
        });
    }
    await model.save('file://'+outputdir);
    fs.writeFileSync(path.join(outputdir,'data.json'),JSON.stringify(this,null,4));
};

TokenNetwork.prototype.load = async function(outputdir){
    let self = this;
    if(fs.existsSync(outputdir)){
        let dataFile = path.join(outputdir,'data.json')
        if(fs.existsSync(dataFile)){
            let options = JSON.parse(fs.readFileSync(dataFile,{encoding:'utf-8'}));
            Object.keys(options).forEach(function(k){
                self[k] = options[k];
            });
        }
        let modelFile = path.join(outputdir,'model.json');
        if(fs.existsSync(modelFile)){
            let model = await tf.loadLayersModel('file://'+modelFile);
            model.compile({
                loss:tf.losses.meanSquaredError,
                optimizer:tf.train.sgd(self.learningRate)          
            });
            self.model = model;
        }
    }
};

TokenNetwork.prototype.weightToToken = function(weight){
    let self = this;
    let charactersCount = self.charactersCount;
    let prod = self.tokenInterval*weight;
    let indexes = [];
    while(prod > 0){
        indexes.push(prod % charactersCount);
        prod = Math.floor(prod / charactersCount);
    }
    return indexes
        .map(index => self.characters[index]?self.characters[index]:null)
        .filter(chr => chr !== null)
        .join('');
};

TokenNetwork.prototype.getNearestToken = function(weight){
    let self = this;
    let index = parseInt(weight)-1;
    if(index >= 0 && self.tokens[index]){
        return self.tokens[index]
    }
    return null;
    /*
    let self = this; 
    let diff = null;
    let nearest = null;
    let tokensWeights = self.tokensWeights;
    Object.keys(tokensWeights).forEach(function(token){
        let w = tokensWeights[token];
        let d  = Math.abs(weight-w);
        if(diff === null || diff > d){
            diff = d;
            nearest = token;
        }
    });
    return nearest;*/
};

TokenNetwork.prototype.getTokenWeight = function(token){
    return this.tokens.indexOf(token)+1;
    /*
    let self = this;
    let charactersCount = self.charactersCount;
    let prod = token.split('').map((chr) => self.characters.indexOf(chr)).reduce(function(prod,current,index){
        return prod+(current*Math.pow(charactersCount,token.length-index-1));
    },0);
    return prod/self.tokenInterval;*/
};


TokenNetwork.prototype.incrementLearningRate = function(){
    let self = this;
    let zeros = 0;
    let tmp = String(self.learningRate).split('.');
    if(tmp[1]){
        zeros = tmp[1].length;
    }
    self.learningRate = 1/Math.pow(10,zeros+1);
    return self;
};



function initialize(self){
    let tokenMaxLength = null;
    let tokens = null;
    let tokensWeights = null;
    let characters = [];
    let learningRate = 0.1;
    let loss = null;
    let letterSeparators = [' '];
    let paragraphSeparators = ['\n'];
    let tokenSeparators = ['.',',',';','!','\'','(',')',':','?'];
    let data = [];
    let tokenInterval = null;
    let lineInterval = null;
    let units = null;
    let model = null;
    let batchSize = 128;

    let reset = function(){
        tokens = null;
        tokensWeights = null;
        tokenMaxLength = null;
        units = null;
        tokenInterval = null;
        lineInterval = null;
    };


    Object.defineProperty(self,'tokenMaxLength',{
        get:function(){
            if(tokenMaxLength === null){
                tokenMaxLength = self.tokens.reduce((s,t) => Math.max(s,t.length),0);
            }
            return tokenMaxLength;
        }
    });

    Object.defineProperty(self,'paragraphs',{
        get:function(){
            let tmp = [];
            data.forEach(function(paragraphs){
                tmp = tmp.concat(paragraphs);
            });
            return tmp;
        }
    });

    Object.defineProperty(self,'tokens',{
        get:function(){
            if(tokens === null){
                tokens = [];
                self.paragraphs.forEach(function(paragraph){
                    paragraph.forEach(function(token){
                        if(tokens.indexOf(token) === -1){
                            tokens.push(token);
                        }
                    });
                });
                tokens = tokens.sort(function(a,b){
                    let diff =  a.length - b.length;
                    if(diff === 0){
                        diff = a.localeCompare(b);
                    }
                    return diff;
                });
            }
            return tokens;
        },
        set:function(tks){
            if(!tks){
                reset();
            }
        }
    });

    Object.defineProperty(self,'tokensWeights',{
        get:function(){
            if(tokensWeights === null){
                let self = this;
                tokensWeights = [];
                self.tokens.forEach(function(token){
                    tokensWeights[token] = self.getTokenWeight(token);
                });
            }
            return tokensWeights;
        }
    });

    Object.defineProperty(self,'characters',{
        get:function(){
            return characters;
        },
        set:function(chrs){
            characters = [...new Set(chrs)].sort((a,b) => a.charCodeAt(0) - b.charCodeAt(0));
            reset();
        }
    });

    Object.defineProperty(self,'charactersCount',{
        get:function(){
            return characters.length;
        }
    });
    

    Object.defineProperty(self,'learningRate',{
        get:function(){
            return learningRate;
        },
        set:function(lr){
            learningRate = lr;
        }
    });

    Object.defineProperty(self,'letterSeparators',{
        get:function(){
            return letterSeparators;
        },
        set:function(ls){
            letterSeparators = [...new Set(ls)].sort();
            reset();
        }
    });

    Object.defineProperty(self,'paragraphSeparators',{
        get:function(){
            return paragraphSeparators;
        },
        set:function(ps){
            paragraphSeparators = [...new Set(ps)].sort();
            reset();
        }
    });

    Object.defineProperty(self,'tokenSeparators',{
        get:function(){
            return tokenSeparators;
        },
        set:function(ts){
            tokenSeparators = [...new Set(ts)].sort();
            reset();
        }
    });

    Object.defineProperty(self,'data',{
        get:function(){
            return data;
        },
        set:function(d){
            data = d;
        }
    });

    Object.defineProperty(self,'batchSize',{
        get:function(){
            return batchSize;
        },
        set:function(b){
            batchSize = Math.max(b,2);            
        }
    });

    Object.defineProperty(self,'trainingData',{
        get:function(){
            let trainingData = [];
            let b = Math.min(batchSize,data.length);
            for(let i = 0; i < data.length-batchSize; i++){
                let sa = i;
                let sb = i+1;
                trainingData.push([
                    self.lineTo3d(data.slice(sa,sa+batchSize)),
                    self.lineTo3d(data.slice(sb,sb+batchSize))
                ]);
            }
            return trainingData;
        }
    });

    Object.defineProperty(self,'tokenInterval',{
        get:function(){
            if(tokenInterval === null){
                tokenInterval = Math.pow(self.charactersCount,self.tokenMaxLength);
            }
            return tokenInterval;
        }
    });

    Object.defineProperty(self,'lineInterval',{
        get:function(){
            if(lineInterval === null){
                let length = self.paragraphs.length;
                let count = 2;
                do{
                    lineInterval = Math.pow(2,count);
                    count++;
                }
                while(lineInterval < length);
            }
            return lineInterval;
        }
    });

    Object.defineProperty(self,'maxParagraphLength',{
        get:function(){
            return self.paragraphs.reduce((a,b) => Math.max(a,b.length),0);
        }
    });

    Object.defineProperty(self,'units',{
        get:function(){
            if(units === null){
                let count = 2;
                let length = self.maxParagraphLength;
                let tmp = 2;
                while(tmp < length){
                    tmp = Math.pow(2,count);
                    count++;
                }
                units = tmp;
            }
            return units;
        }
    });

    Object.defineProperty(self,'model',{
        get:function(){
            if(model === null){
                let m = tf.sequential();
                let rnn = tf.layers.simpleRNN({units:self.batchSize,returnSequences:true,activation:'linear'});
                let inputLayer = tf.input({
                    shape:[self.batchSize,1]
                });
                rnn.apply(inputLayer);
                m.add(rnn);
                m.add(tf.layers.dense({units:1,inputShape:[self.batchSize,self.batchSize],activation:'linear'}));
                m.compile({
                    loss:tf.losses.meanSquaredError,
                    optimizer:tf.train.sgd(self.learningRate)        
                });
                model = m;
            }
            return model;
        },
        set:function(m){
            model = m;
        }
    });

    Object.defineProperty(self,'loss',{
        set:function(l){
            if(!isNaN(l)){
                loss = l;
            }
        },
        get:function(){
            return loss;
        }
    });
}

module.exports = TokenNetwork;