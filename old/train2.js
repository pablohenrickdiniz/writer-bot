const { input } = require('@tensorflow/tfjs-node-gpu');
const tf = require('@tensorflow/tfjs-node-gpu');

function create_model(vocab_size,embedding_dim,rnn_units){
    /*
    let model = tf.sequential({
        layers:[
            tf.layers.embedding({inputDim:vocab_size,outputDim:embedding_dim}),
            tf.layers.gru({
                units:rnn_units,
                returnSequences:true,
                returnState:true
            }),
            tf.layers.dense({units:vocab_size})
        ]
    });
    */


    let embedding = tf.layers.embedding({inputDim:vocab_size,outputDim:embedding_dim});
    let gru =  tf.layers.gru({
        units:rnn_units,
        returnSequences:true
    });
    let dense =  tf.layers.dense({units:vocab_size});   

    let model = tf.sequential();
    model.add(embedding);
    model.add(gru);
    model.add(dense);

    model.compile({
        loss:'meanSquaredError',
        optimizer:tf.train.adam(0.001)     
    });

    model.summary();

    return model;
}



(async function (){
    const fs = require('fs');
   
    const seq_length = 100;
    const BATCH_SIZE = 64;
    const BUFFER_SIZE = 10000;


    function tensor_to_array(tensor){
        return tensor.arraySync();
    }

    function array_to_tensor(array){
        return tf.tensor(array);
    }

    function characters_to_ids(characters,vocab){
        return characters.map(function(character){
            return character_to_id(character,vocab);
        });
    }

    function character_to_id(character,vocab){
        return vocab.indexOf(character);
    }

    function id_to_character(id,vocab){
        return vocab[id];
    }

    function ids_to_characters(ids,vocab){
        return ids.map(function(id){
            return id_to_character(id,vocab);
        });
    }

    function ids_to_text(ids,vocab){
        return characters_to_text(ids_to_characters(ids,vocab));
    }

    function text_to_tensor(text){
        return tf.tensor(text_to_characters(text));
    }

    function text_to_characters(text){
        return text.split("");
    }

    function characters_to_text(characters){
        return characters.join("");
    }

    function text_to_ids(text,vocab){
        return characters_to_ids(text_to_characters(text),vocab);
    }

    function tensor_to_text(tensor){
        return tensor_to_array(tensor).join("");
    }

    function text_to_vocab(text){
        return [...new Set(text)].sort();
    }

    function split_input_target_array(array){
        return [
            array_to_tensor(array.slice(0,array.length-1)),
            array_to_tensor(array.slice(1,array.length)),
        ];
    }

    function split_input_target_tensor(tensor){
        return split_input_target_array(tensor_to_array(tensor));
    }

    function predict_array_to_ids(prediction){
        return prediction.map(function(p){
            let index = 0;
            let max = null;
            for(let i  = 0;i < p[0].length;i++){
                if(max === null || p[0][i] > max){
                    max = p[0][i];
                    index = i;
                }
            }
            return index;
        });
    }

    function predict_to_text(tensor,vocab){
        return ids_to_text( predict_array_to_ids(tensor_to_array(tensor)),vocab);
    }

    /*
    function tensor_chunks(tensor,size){
        size = size || 1;
        let arr = tensor_array(tensor);
        let chunks = [];
        for (let i = 0; i < arr.length; i += size) {
            chunks.push(tf.tensor(arr.slice(i, i + size)));
        }
        return chunks;
    }
    */

    text = fs.readFileSync('./data/shakespeare.txt',{encoding:'utf-8'});
    // length of text is the number of characters in it
    console.log('Length of the text: '+text.length+' characters');
    // Take a look at the first 250 characters in text
    //console.log(text.substring(0,250));
    // The unique characters in the file
    const vocab = text_to_vocab(text);

    console.log(vocab.length+' unique characters');
    let all_ids = text_to_ids(text,vocab);
   
    let ids_dataset = tf.data.array(all_ids);
    let sequences = ids_dataset
        .batch(seq_length+1);

    let dataset = sequences.map(split_input_target_tensor).shuffle(BUFFER_SIZE);



    // Length of the vocabulary in chars
    let vocab_size = vocab.length;

    // The embedding dimension
    let embedding_dim = 256;


    // Number of RNN units
    let rnn_units = 1024

    let model = create_model(vocab_size,embedding_dim,rnn_units);

    await dataset.take(1).forEachAsync(function(d){
        let input_example_batch = d[0];
        let example_batch_predictions = model.predict(input_example_batch);
        let text = predict_to_text(example_batch_predictions,vocab);
        fs.writeFileSync("predict.txt",text);
        //sampled_indices = tf.multinomial(example_batch_predictions, 1);
       // sampled_indices = tf.squeeze(sampled_indices,-1);
       // console.log(sampled_indices);
    });

    await dataset.take(100).forEachAsync(async function(d){
        await model.fit([d[0]],[d[1]],{
            epochs:1000,
            verbose:2
        });
    });

    //let example_batch_predictions = model.predict(input_example_batch);
    //console.log(example_batch_predictions.shape);
})();