"use strict";

function _toConsumableArray(arr) { return _arrayWithoutHoles(arr) || _iterableToArray(arr) || _nonIterableSpread(); }

function _nonIterableSpread() { throw new TypeError("Invalid attempt to spread non-iterable instance"); }

function _iterableToArray(iter) { if (Symbol.iterator in Object(iter) || Object.prototype.toString.call(iter) === "[object Arguments]") return Array.from(iter); }

function _arrayWithoutHoles(arr) { if (Array.isArray(arr)) { for (var i = 0, arr2 = new Array(arr.length); i < arr.length; i++) { arr2[i] = arr[i]; } return arr2; } }

var tf = require('@tensorflow/tfjs-node-gpu');

var path = require('path');

var fs = require('fs');

var TextUtils = require('./TextUtils');

function TokenNetwork(options) {
  var self = this;
  initialize(self);
  options = options || {};
  Object.keys(options).forEach(function (k) {
    self[k] = options[k];
  });
}

TokenNetwork.prototype.loadTextFile = function (filename) {
  var self = this;
  self.loadText(fs.readFileSync(filename, {
    encoding: 'utf-8'
  }));
  return self;
};

TokenNetwork.prototype.loadText = function (text) {
  var self = this;
  text = TextUtils.cleanText(text);
  self.characters = self.characters.concat(TextUtils.charactersFromText(text));
  self.data = TextUtils.split(text, self.paragraphSeparators, true).map(function (paragraph) {
    return TextUtils.split(paragraph, self.tokenSeparators.concat(self.letterSeparators), true);
  });
  self.tokens = null;
  return this;
};

TokenNetwork.prototype.lineTo3d = function (l) {
  var self = this;

  var line = _toConsumableArray(l);

  var units = self.units;

  while (line.length < units) {
    line.push(' ');
  }

  line = self.encodeLine(line);
  return tf.tensor(line, [1, self.units, 1]).arraySync()[0];
};

TokenNetwork.prototype.encodeLine = function (l) {
  var self = this;
  return l.map(function (t) {
    return self.getTokenWeight(t);
  });
};

TokenNetwork.prototype.train = function _callee(epochs, step) {
  var self, trainingData, x, y, i, loss, _loop;

  return regeneratorRuntime.async(function _callee$(_context2) {
    while (1) {
      switch (_context2.prev = _context2.next) {
        case 0:
          self = this;
          trainingData = self.trainingData;
          x = [];
          y = [];

          for (i = 0; i < trainingData.length; i++) {
            x.push(trainingData[i][0]);
            y.push(trainingData[i][1]);
          }

          epochs = epochs || 100;
          loss = 0;

          _loop = function _loop() {
            var model, tx, ty;
            return regeneratorRuntime.async(function _loop$(_context) {
              while (1) {
                switch (_context.prev = _context.next) {
                  case 0:
                    model = self.model;
                    tx = tf.tensor(x);
                    ty = tf.tensor(y);
                    _context.next = 5;
                    return regeneratorRuntime.awrap(model.fit(tx, ty, {
                      epochs: epochs,
                      //  batchSize: 32,
                      verbose: 0,
                      callbacks: {
                        onEpochEnd: function onEpochEnd(logs, b) {
                          loss = b.loss;

                          if (!isNaN(loss) && step) {
                            self.loss = loss;
                            step(logs + 1, epochs, loss);
                          } else {
                            model.stopTraining = true;
                            self.incrementLearningRate();
                            self.model = null;
                          }
                        }
                      }
                    }));

                  case 5:
                  case "end":
                    return _context.stop();
                }
              }
            });
          };

        case 8:
          _context2.next = 10;
          return regeneratorRuntime.awrap(_loop());

        case 10:
          if (isNaN(loss)) {
            _context2.next = 8;
            break;
          }

        case 11:
        case "end":
          return _context2.stop();
      }
    }
  }, null, this);
};

TokenNetwork.prototype.predict = function (text) {};

TokenNetwork.prototype.toJSON = function () {
  var self = this;
  return {
    learningRate: self.learningRate,
    loss: self.loss,
    characters: self.characters,
    letterSeparators: self.letterSeparators,
    paragraphSeparators: self.paragraphSeparators,
    tokenSeparators: self.tokenSeparators
  };
};

TokenNetwork.prototype.save = function _callee2(outputdir) {
  var self, model;
  return regeneratorRuntime.async(function _callee2$(_context3) {
    while (1) {
      switch (_context3.prev = _context3.next) {
        case 0:
          self = this;
          model = self.model;

          if (!fs.existsSync(outputdir)) {
            fs.mkdirSync(outputdir, {
              recursive: true
            });
          }

          _context3.next = 5;
          return regeneratorRuntime.awrap(model.save('file://' + outputdir));

        case 5:
          fs.writeFileSync(path.join(outputdir, 'data.json'), JSON.stringify(this, null, 4));

        case 6:
        case "end":
          return _context3.stop();
      }
    }
  }, null, this);
};

TokenNetwork.prototype.load = function _callee3(outputdir) {
  var self, dataFile, options, modelFile, model;
  return regeneratorRuntime.async(function _callee3$(_context4) {
    while (1) {
      switch (_context4.prev = _context4.next) {
        case 0:
          self = this;

          if (!fs.existsSync(outputdir)) {
            _context4.next = 11;
            break;
          }

          dataFile = path.join(outputdir, 'data.json');

          if (fs.existsSync(dataFile)) {
            options = JSON.parse(fs.readFileSync(dataFile, {
              encoding: 'utf-8'
            }));
            Object.keys(options).forEach(function (k) {
              self[k] = options[k];
            });
          }

          modelFile = path.join(outputdir, 'model.json');

          if (!fs.existsSync(modelFile)) {
            _context4.next = 11;
            break;
          }

          _context4.next = 8;
          return regeneratorRuntime.awrap(tf.loadLayersModel('file://' + modelFile));

        case 8:
          model = _context4.sent;
          model.compile({
            loss: tf.losses.meanSquaredError,
            optimizer: tf.train.sgd(self.learningRate)
          });
          self.model = model;

        case 11:
        case "end":
          return _context4.stop();
      }
    }
  }, null, this);
};

TokenNetwork.prototype.weightToToken = function (weight) {
  var self = this;
  var charactersCount = self.charactersCount;
  var prod = self.tokenInterval * weight;
  var indexes = [];

  while (prod > 0) {
    indexes.push(prod % charactersCount);
    prod = Math.floor(prod / charactersCount);
  }

  return indexes.map(function (index) {
    return self.characters[index] ? self.characters[index] : null;
  }).filter(function (chr) {
    return chr !== null;
  }).join('');
};

TokenNetwork.prototype.getNearestToken = function (weight) {
  var self = this;
  var index = parseInt(weight) - 1;

  if (index >= 0 && self.tokens[index]) {
    return self.tokens[index];
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

TokenNetwork.prototype.getTokenWeight = function (token) {
  var self = this;
  return this.tokens.indexOf(token) + 1;
  /*
  let self = this;
  let charactersCount = self.charactersCount;
  let prod = token.split('').map((chr) => self.characters.indexOf(chr)).reduce(function(prod,current,index){
      return prod+(current*Math.pow(charactersCount,token.length-index-1));
  },0);
  return prod/self.tokenInterval;*/
};

TokenNetwork.prototype.incrementLearningRate = function () {
  var self = this;
  var zeros = 0;
  var tmp = String(self.learningRate).split('.');

  if (tmp[1]) {
    zeros = tmp[1].length;
  }

  self.learningRate = 1 / Math.pow(10, zeros + 1);
  return self;
};

function initialize(self) {
  var tokenMaxLength = null;
  var tokens = null;
  var tokensWeights = null;
  var characters = [];
  var learningRate = 0.1;
  var loss = null;
  var letterSeparators = [' '];
  var paragraphSeparators = ['\n'];
  var tokenSeparators = ['.', ',', ';', '!', '\'', '(', ')', ':', '?'];
  var data = [];
  var tokenInterval = null;
  var lineInterval = null;
  var units = null;
  var model = null;

  var reset = function reset() {
    tokens = null;
    tokensWeights = null;
    tokenMaxLength = null;
    units = null;
    tokenInterval = null;
    lineInterval = null;
  };

  Object.defineProperty(self, 'tokenMaxLength', {
    get: function get() {
      if (tokenMaxLength === null) {
        tokenMaxLength = self.tokens.reduce(function (s, t) {
          return Math.max(s, t.length);
        }, 0);
      }

      return tokenMaxLength;
    }
  });
  Object.defineProperty(self, 'tokens', {
    get: function get() {
      if (tokens === null) {
        tokens = _toConsumableArray(new Set(data.reduce(function (a, b) {
          return a.concat(b);
        }))).sort(function (a, b) {
          var diff = a.length - b.length;

          if (diff === 0) {
            diff = a.toString().localeCompare(b.toString());
          }

          return diff;
        });
      }

      return tokens;
    },
    set: function set(tks) {
      if (!tks) {
        reset();
      }
    }
  });
  Object.defineProperty(self, 'tokensWeights', {
    get: function get() {
      if (tokensWeights === null) {
        var _self = this;

        tokensWeights = [];

        _self.tokens.forEach(function (token) {
          tokensWeights[token] = _self.getTokenWeight(token);
        });
      }

      return tokensWeights;
    }
  });
  Object.defineProperty(self, 'characters', {
    get: function get() {
      return characters;
    },
    set: function set(chrs) {
      characters = _toConsumableArray(new Set(chrs)).sort(function (a, b) {
        return a.charCodeAt(0) - b.charCodeAt(0);
      });
      reset();
    }
  });
  Object.defineProperty(self, 'charactersCount', {
    get: function get() {
      return characters.length;
    }
  });
  Object.defineProperty(self, 'learningRate', {
    get: function get() {
      return learningRate;
    },
    set: function set(lr) {
      learningRate = lr;
    }
  });
  Object.defineProperty(self, 'letterSeparators', {
    get: function get() {
      return letterSeparators;
    },
    set: function set(ls) {
      letterSeparators = _toConsumableArray(new Set(ls)).sort();
      reset();
    }
  });
  Object.defineProperty(self, 'paragraphSeparators', {
    get: function get() {
      return paragraphSeparators;
    },
    set: function set(ps) {
      paragraphSeparators = _toConsumableArray(new Set(ps)).sort();
      reset();
    }
  });
  Object.defineProperty(self, 'tokenSeparators', {
    get: function get() {
      return tokenSeparators;
    },
    set: function set(ts) {
      tokenSeparators = _toConsumableArray(new Set(ts)).sort();
      reset();
    }
  });
  Object.defineProperty(self, 'data', {
    get: function get() {
      return data;
    },
    set: function set(d) {
      data = d;
    }
  });
  Object.defineProperty(self, 'trainingData', {
    get: function get() {
      var trainingData = [];

      for (var i = 0; i < data.length - 1; i++) {
        var input = self.lineTo3d(data[i], 0);
        var output = self.lineTo3d(data[i + 1]);
        trainingData.push([input, output]);
      }

      return trainingData;
    }
  });
  Object.defineProperty(self, 'tokenInterval', {
    get: function get() {
      if (tokenInterval === null) {
        tokenInterval = Math.pow(self.charactersCount, self.tokenMaxLength);
      }

      return tokenInterval;
    }
  });
  Object.defineProperty(self, 'lineInterval', {
    get: function get() {
      if (lineInterval === null) {
        var length = self.paragraphs.length;
        var count = 2;

        do {
          lineInterval = Math.pow(2, count);
          count++;
        } while (lineInterval < length);
      }

      return lineInterval;
    }
  });
  Object.defineProperty(self, 'maxParagraphLength', {
    get: function get() {
      var max = 0;
      var sum = 0;

      for (var i = 0; i < data.length; i++) {
        if (data[i] === "\n") {
          sum = 0;
        } else {
          sum++;
        }

        max = Math.max(max, sum);
      }

      return max;
    }
  });
  Object.defineProperty(self, 'units', {
    get: function get() {
      if (units === null) {
        var count = 2;
        var length = self.maxParagraphLength;
        var tmp = 2;

        while (tmp < length) {
          tmp = Math.pow(2, count);
          count++;
        }

        units = tmp;
      }

      return units;
    }
  });
  Object.defineProperty(self, 'model', {
    get: function get() {
      if (model === null) {
        var m = tf.sequential();
        var _units = self.units;
        var rnn = tf.layers.simpleRNN({
          units: 1,
          returnSequences: true,
          activation: 'linear',
          inputShape: [_units, 1]
        });
        m.add(rnn);
        m.add(tf.layers.dense({
          units: 1,
          inputShape: [_units, 1],
          activation: 'linear'
        }));
        m.compile({
          loss: tf.losses.meanSquaredError,
          optimizer: tf.train.sgd(self.learningRate)
        });
        model = m;
      }

      return model;
    },
    set: function set(m) {
      model = m;
    }
  });
  Object.defineProperty(self, 'loss', {
    set: function set(l) {
      if (!isNaN(l)) {
        loss = l;
      }
    },
    get: function get() {
      return loss;
    }
  });
}

module.exports = TokenNetwork;