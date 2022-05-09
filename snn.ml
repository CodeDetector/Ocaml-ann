
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
type 'a io       = { i: 'a; o: 'a }                                  
type vec         = float array          
type mat         = vec array              
type neuralNet   = { a : vec io; ah : vec; w : mat io; c : mat io }          
let  vector       = Array.init   


(* Create a 2Darray of dimensions m*n , element of array is returned by function  *)
let matrix m n f = vector m (fun i -> vector n (f i))                        

(* Create a neural net with ni -> number of input neurons ; nh->number of hidden neurons ;no-> number of output neurons *)
let neuralNet ni nh no =                  
    let init fi fo = { i = matrix (ni + 1) nh fi; o = matrix nh no fo } in   
    let rand x0 x1 = x0 +. Random.float(x1 -. x0) in                         
    { 
      a = { i = vector (ni + 1) (fun _ -> 1.0); o = vector no (fun _ -> 1.0) };     
      ah = vector nh (fun _ -> 1.0);      
      w = init (fun _ _ -> rand (-0.2) 0.4) (fun _ _ -> rand (-0.2) 0.4);    
      c = init (fun _ _ -> 0.0) (fun _ _ -> 0.0)                             
    }  

let sigmoid x = 1.0 /. (1.0 +. exp(-. x)) 

let sigmoid' y = y *. (1.0 -. y)          

let rec fold2 n f a xs ys =               
    let a = ref a in                      
    for i=0 to n-1 do                     
        a := f !a (xs i) (ys i)           
    done;                                 
    !a 

let dot n xs ys = fold2 n (fun t x y -> t +. x *. y) 0.0 xs ys               
let length      = Array.length            
let get         = Array.get               

let forward net x =                   
    let ni, nh, no = Array.length net.a.i, Array.length net.ah, Array.length net.a.o in        
    assert(length x = ni-1);         
    let ai i = if i < ni-1 then x.(i) else net.a.i.(i) in               
    let ah j = sigmoid(dot ni ai (fun i -> net.w.i.(i).(j))) in              
    let ah   = vector nh ah in            
    let ao k = sigmoid(dot nh (Array.get ah) (fun j -> net.w.o.(j).(k))) in        
    {net with a = { i = vector ni ai; o = vector no ao }; ah = ah }          

let update net inputs =                   
    let ni, nh, no = length net.a.i, length net.ah, length net.a.o in        
    assert(length inputs = ni-1);         
    let ai i = if i < ni-1 then inputs.(i) else net.a.i.(i) in               
    let ah j = sigmoid(dot ni ai (fun i -> net.w.i.(i).(j))) in              
    let ah   = vector nh ah in            
    let ao k = sigmoid(dot nh (get ah) (fun j -> net.w.o.(j).(k))) in        
    {net with a = { i = vector ni ai; o = vector no ao }; ah = ah }          

        
let backPropagate net targets n m =       
    let ni, nh, no = length net.a.i, length net.ah, length net.a.o in        

    assert(length targets = no);          

    let od k   = sigmoid' net.a.o.(k) *. (targets.(k) -. net.a.o.(k)) in     
    let od     = vector no od in          
    let hd j   = sigmoid' net.ah.(j) *. dot no (get od) (fun k -> net.w.o.(j).(k)) in                           
    let hd     = vector nh hd in          
    let co j k = od.(k) *. net.ah.(j) in  
    let wo j k = net.w.o.(j).(k) +. n *. co j k +. m *. net.c.o.(j).(k) in   
    let ci i j = hd.(j) *. net.a.i.(i) in 
    let wi i j = net.w.i.(i).(j) +. n *. ci i j +. m *. net.c.i.(i).(j) in   

let init fi fo = { i = matrix ni nh fi; o = matrix nh no fo } in         
{ net with w = init wi wo; c = init ci co },                             
0.5 *. fold2 no (fun t x y -> t +. (x -. y) ** 2.0) 0.0                  
            (get targets) (get net.a.o) 
    
    
let rec train net patterns iters n m =    
    if iters = 0 then net else            
        let step (net, error) (inputs, targets) =                            
            let net, de = backPropagate (update net inputs) targets n m in   
            net, error +. de in           
        let net, error = Array.fold_left step (net, 0.0) patterns in         
        if iters mod 10000 = 0 then printf "Error: %g:\n%!" error;           
        train net patterns (iters - 1) n m        


let print_array ff print xs =             
    let n = Array.length xs in            
    if n = 0 then fprintf ff "[||]" else begin                               
        fprintf ff "[|";                   
        for i=0 to Array.length xs-2 do    
            fprintf ff "%a; " print xs.(i) 
    done                               
    end
    
let test patts net =                      
    let aux (inputs, _) =                 
        let print ff = print_array ff (fun ff -> fprintf ff "%g") in         
        let outputs = (update net inputs).a.o in                             
        printf "%a -> %a\n" print inputs print outputs in                    
    Array.iter aux patts 
(* Sample dataset  *)
let df =                               
    [|[|2.0; 0.0; 0.0|] , [|2.0|];             
      [|4.0; 1.0; 1.0|] , [|1.0|];             
      [|1.0; 5.0; 0.0|] , [|1.0|];             
      [|1.0; -1.0; 2.0|] , [|4.0|]|]            

let () =                                  
    let t = Sys.time() in                 
    let net = neuralNet 3 2 1 in          
    test df (train net df 10000 0.3 0.1);                             
    printf "Took %gs\n" (Sys.time() -. t)

