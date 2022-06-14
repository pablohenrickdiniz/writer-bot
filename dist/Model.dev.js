"use strict";

var tf = require('@tensorflow/tfjs-node-gpu');

var fs = require('fs');

var path = require('path');

var Sequencializer = require('./Sequencializer');

function Model(options) {
  var self = this;
  initialize(self, options);
}

function sample(probs, temperature) {
  return tf.tidy(function () {
    var logits = tf.div(tf.log(probs), Math.max(temperature, 1e-6));
    var isNormalized = false;
    return tf.multinomial(logits, 1, null, isNormalized).dataSync()[0];
  });
}

function initialize(self, options) {
  options = options || {};
  var seqLength = options.seqLength || 128;
  var hiddenLayers = options.hiddenLayers || 1;
  var units = options.units || 128;
  var model = null;
  var sequencializer = null;
  var learningRate = 0.001;

  var train = function train(epochs, callback) {
    return regeneratorRuntime.async(function train$(_context) {
      while (1) {
        switch (_context.prev = _context.next) {
          case 0:
            _context.next = 2;
            return regeneratorRuntime.awrap(self.model.fitDataset(self.sequencializer.randomSequences, {
              epochs: epochs,
              verbose: 0,
              callbacks: {
                onEpochEnd: function onEpochEnd(epochs, log) {
                  callback(epochs + 1, log.loss);
                }
              }
            }));

          case 2:
          case "end":
            return _context.stop();
        }
      }
    });
  };

  var generate = function generate(length, callback) {
    var model, indexes, text, count, xBuffer, i, input, output, index, chr;
    return regeneratorRuntime.async(function generate$(_context2) {
      while (1) {
        switch (_context2.prev = _context2.next) {
          case 0:
            model = self.model;
            _context2.next = 3;
            return regeneratorRuntime.awrap(self.sequencializer.dataset.batch(seqLength).map(function (t) {
              return t.arraySync();
            }).take(1).toArray());

          case 3:
            indexes = _context2.sent[0];
            text = indexes.map(function (i) {
              return self.sequencializer.decodeIndex(i);
            }).join("");

            if (callback) {
              callback(text);
            }

            count = 0;

            do {
              xBuffer = tf.buffer([1, seqLength, self.sequencializer.encoderSize]);

              for (i = 0; i < seqLength; i++) {
                xBuffer.set(1, 0, i, indexes[i]);
              }

              input = xBuffer.toTensor();
              /** tensor de probabilidades */

              output = model.predict(input).squeeze();
              index = sample(output, 0.6);
              chr = sequencializer.decodeIndex(index);
              text += chr;

              if (callback) {
                callback(chr);
              }

              indexes.shift();
              indexes.push(index);
              count++;
            } while (count < length);

            return _context2.abrupt("return", text);

          case 9:
          case "end":
            return _context2.stop();
        }
      }
    });
  };

  var loadTextFile = function loadTextFile(filename) {
    self.sequencializer.loadTextFile(filename);
    return self;
  };

  var loadText = function loadText(text) {
    self.sequencializer.loadText(text);
    return self;
  };

  var save = function save(outputdir) {
    var model;
    return regeneratorRuntime.async(function save$(_context3) {
      while (1) {
        switch (_context3.prev = _context3.next) {
          case 0:
            model = self.model;

            if (!fs.existsSync(outputdir)) {
              fs.mkdirSync(outputdir, {
                recursive: true
              });
            }

            _context3.next = 4;
            return regeneratorRuntime.awrap(model.save('file://' + outputdir));

          case 4:
            fs.writeFileSync(path.join(outputdir, 'data.json'), JSON.stringify(this, null, 4));

          case 5:
          case "end":
            return _context3.stop();
        }
      }
    }, null, this);
  };

  var load = function load(outputdir) {
    var dataFile, modelFile, tmp;
    return regeneratorRuntime.async(function load$(_context4) {
      while (1) {
        switch (_context4.prev = _context4.next) {
          case 0:
            if (fs.existsSync(outputdir)) {
              _context4.next = 2;
              break;
            }

            return _context4.abrupt("return", false);

          case 2:
            dataFile = path.join(outputdir, 'data.json');

            if (fs.existsSync(dataFile)) {
              _context4.next = 5;
              break;
            }

            return _context4.abrupt("return", false);

          case 5:
            modelFile = path.join(outputdir, 'model.json');

            if (fs.existsSync(modelFile)) {
              _context4.next = 8;
              break;
            }

            return _context4.abrupt("return", false);

          case 8:
            _context4.next = 10;
            return regeneratorRuntime.awrap(tf.loadLayersModel('file://' + modelFile));

          case 10:
            model = _context4.sent;
            model.compile({
              optimizer: tf.train.rmsprop(learningRate),
              loss: 'categoricalCrossentropy'
            });
            tmp = JSON.parse(fs.readFileSync(dataFile, {
              encoding: 'utf-8'
            }));
            Object.keys(tmp).forEach(function (k) {
              switch (k) {
                case 'learningRate':
                  learningRate = tmp[k];
                  break;

                case 'hiddenLayers':
                  hiddenLayers = tmp[k];
                  break;

                case 'units':
                  units = tmp[k];
                  break;

                case 'sequencializer':
                  sequencializer = new Sequencializer(tmp[k]);
                  break;
              }
            });

          case 14:
          case "end":
            return _context4.stop();
        }
      }
    });
  };

  var toJSON = function toJSON() {
    return {
      learningRate: self.learningRate,
      sequencializer: self.sequencializer,
      hiddenLayers: hiddenLayers,
      units: units
    };
  };

  Object.defineProperty(self, 'seqLength', {
    get: function get() {
      return seqLength;
    }
  });
  Object.defineProperty(self, 'model', {
    get: function get() {
      if (model === null) {
        model = tf.sequential();
        var seq = self.sequencializer;

        for (var i = 0; i < hiddenLayers; ++i) {
          model.add(tf.layers.lstm({
            units: units,
            returnSequences: i < hiddenLayers - 1,
            inputShape: i === 0 ? [seqLength, seq.encoderSize] : undefined
          }));
        }

        model.add(tf.layers.dense({
          units: seq.encoderSize,
          activation: 'softmax'
        }));
        model.compile({
          optimizer: tf.train.rmsprop(learningRate),
          loss: 'categoricalCrossentropy'
        });
      }

      return model;
    }
  });
  Object.defineProperty(self, 'train', {
    get: function get() {
      return train;
    }
  });
  Object.defineProperty(self, 'generate', {
    get: function get() {
      return generate;
    }
  });
  Object.defineProperty(self, 'sequencializer', {
    get: function get() {
      if (sequencializer === null) {
        sequencializer = new Sequencializer({
          type: options.encoder,
          seqLength: seqLength
        });
      }

      return sequencializer;
    }
  });
  Object.defineProperty(self, 'loadText', {
    get: function get() {
      return loadText;
    }
  });
  Object.defineProperty(self, 'loadTextFile', {
    get: function get() {
      return loadTextFile;
    }
  });
  Object.defineProperty(self, 'save', {
    get: function get() {
      return save;
    }
  });
  Object.defineProperty(self, 'load', {
    get: function get() {
      return load;
    }
  });
  Object.defineProperty(self, 'learningRate', {
    get: function get() {
      return learningRate;
    }
  });
  Object.defineProperty(self, 'toJSON', {
    get: function get() {
      return toJSON;
    }
  });
}

module.exports = Model;