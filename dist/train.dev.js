"use strict";

var TokenNetwork = require('./TokenNetwork');

var fs = require('fs');

var path = require('path');

var TextUtils = require('./TextUtils');

var dataDir = './data';

function init() {
  var network;
  return regeneratorRuntime.async(function init$(_context) {
    while (1) {
      switch (_context.prev = _context.next) {
        case 0:
          network = new TokenNetwork();
          _context.next = 3;
          return regeneratorRuntime.awrap(network.load('./models/resumo'));

        case 3:
          network.loadTextFile('./data/resumo.txt');

        case 4:
          if (!true) {
            _context.next = 11;
            break;
          }

          fs.writeFileSync('resumo.txt', network.predict('A terra era sem forma e va'));
          _context.next = 8;
          return regeneratorRuntime.awrap(network.train(100, function (logs, epochs, loss) {
            var log = logs.toString().padStart(epochs.toString().length, '0') + '/' + epochs + ' - treinando (loss:' + loss + ')';
            console.log(log);
          }));

        case 8:
          network.save('./models/resumo');
          _context.next = 4;
          break;

        case 11:
        case "end":
          return _context.stop();
      }
    }
  });
}

init();