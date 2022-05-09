
(*In this implementation of an Artificial Neural Network(ANN) ,we are using a sample dataset consisting of 3 features and 1 target   
The ocaml code comprises maily of 5 functions - 
    1) neuralNet - initialises the neural network with some inital values for weights,bias and neuron values of each layer .
    2) train - Performs training on the dataset . It is a function that contains 2 functions - 1) forward 2) backward. 
    3) forward - This function passes the data through the network each time it is called.
    4) backward - After forward feed , we find the loss and perform gradient descent by calculating del(L)/del(w) for each weight matrix . This function returns the network and the error in the model prediction . 
    5) test - Finally we perform predictions on our trained model .

Some other functions used are - 
    1) dot - Calculate dot product of 2 matrices . (It uses a recursive function fold2 for looping through all the rows and columns )
    2)sigmoid - Perform sigmoid activation on the output of each layer in the ANN . 
    3)matrix - Forms 2d array . 

 *)

open Printf       
(* descriptions of input and output layer                 *)
type 'a io       = { i: 'a; o: 'a }                                  
type vec         = float array          
type mat         = vec array              
type neuralNet   = { a : vec io; ah : vec; w : mat io; c : mat io }          
let  vector      = Array.init   
let  length      = Array.length            
let  get         = Array.get  

(* Create a 2Darray of dimensions m*n , element of array is returned by function  *)
let matrix m n f = vector m (fun i -> vector n (f i))                        

(* Create a neural net with ni -> number of input neurons ; nh->number of hidden neurons ;no-> number of output neurons *)
let neuralNet ni nh no =                  
    let init fi fo = { i = matrix (ni + 1) nh fi; o = matrix nh no fo } in   
    let rand x0 x1 = x0 +. Random.float(x1 -. x0) in                         
    { 
        (* here we are initializing inputs, outputs, weights and biases *)
      a = { i = vector (ni + 1) (fun _ -> Random.float 0.2); o = vector no (fun _ -> Random.float 0.2) };     
      ah = vector nh (fun _ -> Random.float 0.2);      
      w = init (fun _ _ -> rand (-0.2) 0.4) (fun _ _ -> rand (-0.2) 0.4);    
      c = init (fun _ _ -> 0.2) (fun _ _ -> 0.2)                             
    }  

let sigmoid x = 1.0 /. (1.0 +. exp(-. x)) 

(* 
    sigmoid(x)' = sigmoid(x)* (1- sigmoid(x))
*)
let sigmoid' y = y *. (1.0 -. y)          

(* Calculating dot product of two vectors  *)
(* Higher order function  *)
let rec fold2 n f a xs ys =               
    let a = ref a in                      
    for i=0 to n-1 do                     
        a := f !a (xs i) (ys i)           
    done;                                 
    !a 

let dot n xs ys = fold2 n (fun t x y -> t +. x *. y) 0.0 xs ys               
             
(* Foward propogation *)
let forward net x =                   
    let ni, nh, no = length net.a.i, length net.ah, length net.a.o in        
    assert(length x = ni-1);  
    (* Storing input in input layer 
        Input layer takes each row as input
    *)
    let ai i = if i < ni-1 then x.(i) else net.a.i.(i) in   
    (* Hidden layer pass with sigmoid activation function  *)
    let ah j = sigmoid(dot ni ai (fun i -> net.w.i.(i).(j))) in              
    let ah   = vector nh ah in            
    (* Output layer pass with sigmoid activation function  *)
    let ao k = sigmoid(dot nh (get ah) (fun j -> net.w.o.(j).(k))) in        
    {net with a = { i = vector ni ai; o = vector no ao }; ah = ah }          

         
let backPropagate net targets n  =       
    let ni, nh, no = length net.a.i, length net.ah, length net.a.o in        

    assert(length targets = no);          
    (* Partial Derivative of output layer *)
    let od k   = sigmoid' net.a.o.(k) *. (targets.(k) -. net.a.o.(k)) in     
    let od     = vector no od in 
    (* Partial Derivative of hidden layer *)
    let hd j   = sigmoid' net.ah.(j) *. dot no (get od) (fun k -> net.w.o.(j).(k)) in                           
    let hd     = vector nh hd in   
    (* Updating weights : wo -> output weights ; wi -> input weights*)
    let co j k = od.(k) *. net.ah.(j) in  
    let wo j k = net.w.o.(j).(k) +. n *. co j k +. n *. net.c.o.(j).(k) in   
    let ci i j = hd.(j) *. net.a.i.(i) in 
    let wi i j = net.w.i.(i).(j) +. n *. ci i j +. n *. net.c.i.(i).(j) in   

    let init fi fo = { i = matrix ni nh fi; o = matrix nh no fo } in         
    { net with w = init wi wo; c = init ci co },                             
    0.05 *. fold2 no (fun t x y -> t +. (x -. y) ** 2.0) 0.0                  
                (get targets) (get net.a.o) 
    
    
let rec train net patterns iters n  =    
    if iters = 0 then net else            
        let step (net, error) (inputs, targets) =                            
            let net, de = backPropagate (forward net inputs) targets n  in   
            net, error +. de in           
        let net, error = Array.fold_left step (net, 0.0) patterns in         
        if iters mod 1000 = 0 then printf "Error: %g:\n%!" error;           
        train net patterns (iters - 1) n         


let print_array ff print xs =             
    let n = Array.length xs in            
    if n = 0 then fprintf ff "[||]" else begin                               
        for i=0 to Array.length xs-1 do    
            fprintf ff "%a; " print xs.(i)
        done                       
    end
    
let test patts net =                      
    let aux (inputs, _) =                 
        let print ff = print_array ff (fun ff -> fprintf ff "%g") in         
        let outputs = (forward net inputs).a.o in                             
        printf "%a -> %a\n" print inputs print outputs in                    
    Array.iter aux patts 
(* Sample dataset  *)
let df =                               
    [|[|2.0; 0.0; 0.0|] , [|2.0|];             
      [|4.0; 1.0; 1.0|] , [|1.0|];             
      [|1.0; 5.0; 0.0|] , [|1.0|];             
      [|1.5; -1.0; 2.0|] , [|4.0|];
      [|2.0; 0.05; 0.25|] , [|2.0|];             
      [|10.0; 10.0; 12.0|] , [|3.0|];             
      [|12.0; 3.0; 9.0|] , [|5.0|];             
      [|9.0; -5.0; 2.0|] , [|4.0|]|]            

let () =                                  
    let t = Sys.time() in         
    (* 3 input neurons, 2 hidden neurons, 1 output neuron *)
    let net = neuralNet 3 2 1 in     
    (* model, number of iterations, learning rate      *)
    test df (train net df 10000 0.01);                             
    printf "Took %gs\n" (Sys.time() -. t)

