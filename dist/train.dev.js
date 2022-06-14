"use strict";

var tf = require('@tensorflow/tfjs-node-gpu');

var fs = require('fs');

var modelDir = '/content/drive/MyDrive/ia-projects/writer-bot/models/biblia';
var outputFile = '/content/drive/MyDrive/ia-projects/writer-bot/output/biblia.txt';

(function _callee2() {
  var Model, model, loaded, epochs;
  return regeneratorRuntime.async(function _callee2$(_context2) {
    while (1) {
      switch (_context2.prev = _context2.next) {
        case 0:
          Model = require('./Model');
          model = new Model({
            encoder: 'vocab'
          });
          loaded = false;
          _context2.prev = 3;
          _context2.next = 6;
          return regeneratorRuntime.awrap(model.load(modelDir));

        case 6:
          loaded = _context2.sent;
          _context2.next = 11;
          break;

        case 9:
          _context2.prev = 9;
          _context2.t0 = _context2["catch"](3);

        case 11:
          if (!loaded) {
            model.loadTextFile('./data/biblia.txt');
          }

          epochs = 100;
          _context2.next = 15;
          return regeneratorRuntime.awrap(model.train(epochs, function _callee(index, loss) {
            var res;
            return regeneratorRuntime.async(function _callee$(_context) {
              while (1) {
                switch (_context.prev = _context.next) {
                  case 0:
                    _context.next = 2;
                    return regeneratorRuntime.awrap(model.save(modelDir));

                  case 2:
                    console.log(index + '/' + epochs + ' - treinando, taxa de erro:' + loss.toFixed(8));
                    res = fs.createWriteStream(outputFile, 'utf-8');
                    _context.next = 6;
                    return regeneratorRuntime.awrap(model.generate(8192, function (t) {
                      res.write(t);
                    }));

                  case 6:
                    res.close();

                  case 7:
                  case "end":
                    return _context.stop();
                }
              }
            });
          }));

        case 15:
        case "end":
          return _context2.stop();
      }
    }
  }, null, null, [[3, 9]]);
})();