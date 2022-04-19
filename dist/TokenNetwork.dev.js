"use strict";

function _toConsumableArray(arr) { return _arrayWithoutHoles(arr) || _iterableToArray(arr) || _nonIterableSpread(); }

function _nonIterableSpread() { throw new TypeError("Invalid attempt to spread non-iterable instance"); }

function _iterableToArray(iter) { if (Symbol.iterator in Object(iter) || Object.prototype.toString.call(iter) === "[object Arguments]") return Array.from(iter); }

function _arrayWithoutHoles(arr) { if (Array.isArray(arr)) { for (var i = 0, arr2 = new Array(arr.length); i < arr.length; i++) { arr2[i] = arr[i]; } return arr2; } }

var tf = require('@tensorflow/tfjs-node-gpu');

var path = require('path');

var fs = require('fs');

var TextUtils = require('./TextUtils');

var _require = require('@tensorflow/tfjs-node-gpu'),
    atanh = _require.atanh;

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
  var contents = fs.readFileSync(filename, {
    encoding: 'utf-8'
  });
  self.loadText(contents);
  return self;
};

TokenNetwork.prototype.loadText = function (text) {
  var self = this;
  text = TextUtils.cleanText(text);
  self.characters = self.characters.concat(TextUtils.charactersFromText(text));
  var paragraphs = TextUtils.split(text, self.paragraphSeparators).map(function (paragraph) {
    var tokens = [];
    TextUtils.split(paragraph, self.letterSeparators).forEach(function (w) {
      tokens = tokens.concat(TextUtils.split(w, self.tokenSeparators, true));
    });
    return tokens;
  });
  self.data.push(paragraphs);
  self.tokens = null;
  return this;
};

TokenNetwork.prototype.getCharactersCount = function () {
  return this.characters.length;
};

TokenNetwork.prototype.getModel = function _callee() {
  var self, model, units;
  return regeneratorRuntime.async(function _callee$(_context) {
    while (1) {
      switch (_context.prev = _context.next) {
        case 0:
          self = this;

          if (!(self.model === null)) {
            _context.next = 13;
            break;
          }

          if (!fs.existsSync(self.getModelJsonFile())) {
            _context.next = 8;
            break;
          }

          _context.next = 5;
          return regeneratorRuntime.awrap(tf.loadLayersModel('file://' + self.getModelJsonFile()));

        case 5:
          model = _context.sent;
          _context.next = 11;
          break;

        case 8:
          units = self.units;
          model = tf.sequential();
          model.addLayer(tf.layers.dense({
            units: units,
            inputShape: [1, 2]
          }));

        case 11:
          model.compile({
            loss: 'meanSquaredError',
            optimizer: tf.train.sgd(self.getLearningRate())
          });
          self.model = model;

        case 13:
          return _context.abrupt("return", self.model);

        case 14:
        case "end":
          return _context.stop();
      }
    }
  }, null, this);
};

TokenNetwork.prototype.train = function _callee2(epochs) {
  var self, trainingData, x, y, i, t, loss, _loop;

  return regeneratorRuntime.async(function _callee2$(_context3) {
    while (1) {
      switch (_context3.prev = _context3.next) {
        case 0:
          self = this;
          trainingData = self.trainingData;
          x = [];
          y = [];

          for (i = 0; i < trainingData.length; i++) {
            t = trainingData[i];
            x.push(t[0]);
            y.push(t[1]);
          }

          epochs = epochs || 100;
          loss = 0;

          _loop = function _loop() {
            var model, tx, ty;
            return regeneratorRuntime.async(function _loop$(_context2) {
              while (1) {
                switch (_context2.prev = _context2.next) {
                  case 0:
                    self.model = null;
                    _context2.next = 3;
                    return regeneratorRuntime.awrap(self.getModel());

                  case 3:
                    model = _context2.sent;
                    tx = tf.tensor(x);
                    ty = tf.tensor(y);
                    _context2.next = 8;
                    return regeneratorRuntime.awrap(model.fit(tx, ty, {
                      epochs: epochs,
                      verbose: 0,
                      callbacks: {
                        onEpochEnd: function onEpochEnd(logs, b) {
                          loss = b.loss;

                          if (logs % 1 === 0) {
                            var log = [[logs.toString().padStart(epochs.toString().length, '0'), epochs].join('/'), 'treinando ' + self.name + ', loss:' + b.loss].join(' - ');

                            if (!isNaN(loss)) {
                              console.log(log);
                            }
                          }

                          if (isNaN(loss)) {
                            model.stopTraining = true;
                            self.incrementLearningRate();
                          }
                        }
                      }
                    }));

                  case 8:
                  case "end":
                    return _context2.stop();
                }
              }
            });
          };

        case 8:
          _context3.next = 10;
          return regeneratorRuntime.awrap(_loop());

        case 10:
          if (isNaN(loss)) {
            _context3.next = 8;
            break;
          }

        case 11:
          if (isNaN(loss)) {
            _context3.next = 14;
            break;
          }

          _context3.next = 14;
          return regeneratorRuntime.awrap(self.save());

        case 14:
        case "end":
          return _context3.stop();
      }
    }
  }, null, this);
};

TokenNetwork.prototype.predict = function _callee3(lines) {
  var _this = this;

  var self, model, result;
  return regeneratorRuntime.async(function _callee3$(_context4) {
    while (1) {
      switch (_context4.prev = _context4.next) {
        case 0:
          self = this;
          lines = lines.constructor !== [].constructor ? [lines] : lines;
          _context4.next = 4;
          return regeneratorRuntime.awrap(self.getModel());

        case 4:
          model = _context4.sent;
          lines = lines.map(function (l) {
            return _this.encodeLine(l);
          });
          result = model.predict(tf.tensor(lines, [lines.length, 1])).arraySync();
          return _context4.abrupt("return", result.map(function (encoded) {
            return self.decodeText(encoded);
          }).join("\n"));

        case 8:
        case "end":
          return _context4.stop();
      }
    }
  }, null, this);
};

TokenNetwork.prototype.save = function _callee4() {
  var self, model, modelDir;
  return regeneratorRuntime.async(function _callee4$(_context5) {
    while (1) {
      switch (_context5.prev = _context5.next) {
        case 0:
          self = this;
          _context5.next = 3;
          return regeneratorRuntime.awrap(self.getModel());

        case 3:
          model = _context5.sent;
          modelDir = self.getModelDir();

          if (!fs.existsSync(modelDir)) {
            fs.mkdirSync(modelDir, {
              recursive: true
            });
          }

          _context5.next = 8;
          return regeneratorRuntime.awrap(model.save('file://' + modelDir));

        case 8:
          self.saveLearningRate();

        case 9:
        case "end":
          return _context5.stop();
      }
    }
  }, null, this);
};

TokenNetwork.prototype.getTokenCode = function (token) {
  var self = this;
  var charactersCount = self.charactersCount;
  var prod = token.split('').map(function (chr) {
    return self.characters.indexOf(chr);
  }).reduce(function (prod, current, index) {
    return prod + current * Math.pow(charactersCount, token.length - index - 1);
  }, 0);
  return prod / self.tokenInterval;
};

function initialize(self) {
  var tokenMaxLength = null;
  var tokens = null;
  var characters = [];
  var learningRate = 0.001;
  var letterSeparators = [' '];
  var paragraphSeparators = ['\n'];
  var tokenSeparators = ['.', ',', ';', '!', '\'', '(', ')', ':', '?'];
  var data = [];
  var tokenInterval = null;
  var units = null;

  var reset = function reset() {
    tokens = null;
    tokenMaxLength = null;
    units = null;
    tokenInterval = null;
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
  Object.defineProperty(self, 'paragraphs', {
    get: function get() {
      var tmp = [];
      data.forEach(function (paragraphs) {
        tmp = tmp.concat(paragraphs);
      });
      return tmp;
    }
  });
  Object.defineProperty(self, 'tokens', {
    get: function get() {
      if (tokens === null) {
        tokens = [];
        self.paragraphs.forEach(function (paragraph) {
          paragraph.forEach(function (token) {
            if (tokens.indexOf(token) === -1) {
              tokens.push(token);
            }
          });
        });
        tokens = tokens.sort(function (a, b) {
          var diff = a.length - b.length;

          if (diff === 0) {
            diff = a.localeCompare(b);
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
      learningRate = Math.abs(lr);
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
    }
  });
  Object.defineProperty(self, 'trainingData', {
    get: function get() {
      var trainingData = [];
      var units = self.units;
      data.forEach(function (phrs) {
        trainingData = trainingData.concat(phrs.map(function (tks) {
          tks = tks.map(function (t) {
            return self.getTokenCode(t);
          });

          while (tks.length < units) {
            tks.push(-1);
          }

          return tks;
        }));
      });
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
  Object.defineProperty(self, 'maxParagraphLength', {
    get: function get() {
      return self.paragraphs.reduce(function (a, b) {
        return Math.max(a, b.length);
      }, 0);
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
}

module.exports = TokenNetwork;