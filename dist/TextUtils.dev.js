"use strict";

function cleanWord(word) {
  return word.trim().split('').filter(function (chr) {
    return chr.length > 0;
  }).join('');
}

function cleanLine(text) {
  return text.trim().split(/\s+/).map(function (word) {
    return cleanWord(word);
  }).filter(function (word) {
    return word.length > 0;
  }).join(' ');
}

function cleanText(text) {
  return text.trim().split(/[\r\n]+/).map(function (line) {
    return cleanLine(line);
  }).filter(function (l) {
    return l.length > 0;
  }).join("\n");
}

function charcodesFromText(text) {
  return text.split('').map(function (c) {
    return c.charCodeAt(0);
  });
}

function textFromCharcodes(charcodes) {
  return charcodes.map(function (c) {
    return String.fromCharCode(c);
  }).join('');
}

function encodeText(text, divider) {
  return charcodesFromText(text).map(function (c) {
    return c / divider;
  });
}

function decodeText(encoded, multiplier) {
  return textFromCharcodes(encoded.map(function (charcode) {
    return Math.round(charcode * multiplier);
  }));
}

function charactersFromText(text) {
  var chrs = [];

  for (var i = 0; i < text.length; i++) {
    var chr = text.charAt(i);

    if (chrs.indexOf(chr) === -1) {
      chrs.push(chr);
    }
  }

  return chrs.sort(function (a, b) {
    return a.localeCompare(b);
  });
}

function insertBetween(arr, value) {
  var tmp = [];

  for (var i = 0; i < arr.length; i++) {
    tmp.push(arr[i]);

    if (i < arr.length - 1) {
      tmp.push(value);
    }
  }

  return tmp;
}

function split(text, separator) {
  var preserveSeparators = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : false;
  separator = separator || '';

  if (separator.constructor != [].constructor) {
    separator = [separator];
  }

  for (var i = 0; i < separator.length; i++) {
    var sep = separator[i];

    if (typeof text === 'string') {
      text = text.split(sep);

      if (preserveSeparators) {
        text = insertBetween(text, sep);
      }

      text = text.filter(function (t) {
        return t.length > 0;
      });
    } else {
      var tmp = [];

      for (var j = 0; j < text.length; j++) {
        var _split = text[j].split(sep);

        if (preserveSeparators) {
          _split = insertBetween(_split, sep);
        }

        tmp = tmp.concat(_split);
      }

      text = tmp.filter(function (t) {
        return t.length > 0;
      });
    }
  }

  return text;
}

var TextUtils = {
  cleanText: cleanText,
  cleanLine: cleanLine,
  cleanWord: cleanWord,
  charcodesFromText: charcodesFromText,
  textFromCharcodes: textFromCharcodes,
  encodeText: encodeText,
  decodeText: decodeText,
  charactersFromText: charactersFromText,
  split: split
};
module.exports = TextUtils;