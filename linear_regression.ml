(* Random float list generator  *)
module RandomListGenerator =
struct
  (* Append a number n in the end of the list l *)
  let rec append l n =
    match l with
    | [] -> [n]
    | h :: t -> h :: (append t n)

  (* Generate a list l with size random numbers *)
  let rec gen size l =
    if size = 0 then
      l
    else
      let n = float_of_int (Random.int 1000000 mod 10 ) in
      let list = append l n in
      gen (size - 1) list
end


(* Defining user made dataset *)


(* function that will generate the random data *)
(* let generate_data () =
	let x = [2.3;4.5;2.;1.222;3.] and 
	y = [5.1;3.;1.;1.2;4.3] in 
  x, y *)

let generate_data () =
	let x = RandomListGenerator.gen 10 [] and 
	y = RandomListGenerator.gen 10 [] in 
	x, y

(* Hyperparameter *)
let learning_rate = 0.001

let input_features, target = generate_data()


(*
 		y = m*x + c; 
		m->slope
		c-> y-intercept
*)

(* training model will nead the feature space and target variable *)
let training inputs target = 
  let rec func inputs target numOfiterations m c i = 
  match numOfiterations with 
  | 0 -> [m;c]
  | n ->
    let x = List.nth inputs i in 
    let y_pred = x*.m +. c and trgt = (List.nth target i) in 
    let error_difference = trgt -. y_pred in 
    let m' = m +. learning_rate *. x *. error_difference in 
    let c' = c +. learning_rate *. error_difference in
    func inputs target (n-1) m' c' ((i+1) mod (List.length inputs))
  in 
  func inputs target 300 0. 0. 0
	
	(*
	 feature space, 
	 target variable,
	 number of iterations,
	 initialized value of m, 
	 initialized value of c, 
	 0th iteration
	*)