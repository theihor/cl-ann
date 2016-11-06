(defpackage :cl-ann
  (:use :common-lisp :anaphora :cl-ann/random :cl-ann/transfer-functions))

(in-package :cl-ann)

(defclass neuron ()
  ((bias :initarg :bias
         :initform 1.0
         :accessor n-bias)
   (transfer-function
    :initarg :transfer-function
    :initform #'identity
    :accessor n-func)
   (input-weights
    :initarg :weights
    :initform #()
    :accessor n-weights
    :type simple-vector)))

(defmethod print-object ((n neuron) s)
  (with-slots (bias transfer-function input-weights) n 
    (format s "#<NEURON W = ~A B = ~A F = ~A>"
            input-weights bias transfer-function)))

(defgeneric activate (n input))

(defmethod activate ((n neuron) input)
  (with-accessors ((b n-bias) (f n-func) (ws n-weights)) n
    (funcall f (+ b (loop
                       for x across input
                       for w across ws
                       sum (* w x))))))

(defclass layer ()
  ((neurons :initarg :neurons 
            :accessor l-neurons
            :type list)
   (size :initarg :size
         :accessor l-size)))

(defmethod print-object ((l layer) s)
  (with-slots (size neurons) l
    (format s "#<LAYER ~A ~A>" size neurons)))

(defgeneric propagate-forward (l input))

(defmethod propagate-forward ((l layer) input)
  (with-accessors ((s l-size) (neurons l-neurons)) l
    (loop with array = (make-array s :element-type 'single-float)
       for i from 0 to s
       for n in neurons
       do (setf (aref array i) (activate n input))
       finally (return array))))

(defclass network ()
  ((input-n :initarg :input-n
            :initform 0
            :accessor input-n)
   (output-n :initarg :output-n
             :initform 0
             :accessor output-n)
   (layers :initarg :layers
           :accessor network-layers)))

(defmethod print-object ((n network) s)
  (with-slots (layers) n 
    (format s "#<NETWORK ~{~A~^~%          ~}>" layers)))

(defmethod propagate-forward ((net network) input)
  (loop
     for l in (network-layers net)
     for output = input then (propagate-forward l output)
     finally (return output)))

(defun network->dot (net s)
  (with-slots (forward-links layers) net
    (format s "digraph g {~%")
    (let ((i 0)
          (n->i (make-hash-table :test #'eq)))
      (labels ((%i (n)
                 (aif (gethash n n->i)
                      it
                      (setf (gethash n n->i) (incf i)))))
        (let ((input-layer (loop repeat (length (n-weights (car (l-neurons (car layers)))))
                              collect (gensym)))) 
          (loop for (l1 l2 . rest) on (cons input-layer layers) by #'cdr
             when l2 do
               (loop
                  for n1 in (if (typep l1 'layer)
                                (l-neurons l1)
                                l1)
                  for i from 0
                  do (loop for n2 in (l-neurons l2)
                        unless (= 0 (aref (n-weights n2) i)) 
                        do (format s "    ~A -> ~A [label=\"~1,2F\"];~%"
                                   (%i n1) (%i n2) (aref (n-weights n2) i))))))))
    (format s "}~%")))

(defun make-random-net (in-n out-n other-n &key (transfer-f #'log-sigmoid) hidden-layers-n) 
  (labels ((%b () (random 1.0))
           (%w () (random 1.0))
           (%weights (n)
             (loop with array = (make-array n)
                for i from 0 below n
                do (setf (aref array i) (%w))
                finally (return array)))
           (%neuron ()
             (make-instance 'neuron
                            :bias (%b) 
                            :transfer-function transfer-f))
           (%take (n lst &optional acc)
             (if (or (null lst)
                     (= 0 n))
                 acc
                 (%take (1- n) (cdr lst) (cons (car lst) acc)))))
    (let ( ;; (in-ns (loop repeat in-n collect (%neuron)))
          (out-ns (loop repeat out-n collect (%neuron)))
          (other-ns (loop repeat other-n collect (%neuron)))
          (hidden-layers-n (or hidden-layers-n
                               (random-int :from 1 :to (1+ (round (/ other-n 2))))))
          (net (make-instance 'network
                              :input-n in-n
                              :output-n out-n)))
      (let ((hidden-layers
             (loop for n = hidden-layers-n then (1- n) 
                while (> n 0)
                when (= n 1) collect (make-instance 'layer
                                                    :neurons other-ns 
                                                    :size (length other-ns)) 
                else collect (let* ((s (random-int :from 1 :to (- (length other-ns) n)))
                                    (neurons (%take s other-ns)))
                               (setf other-ns (nthcdr s other-ns))
                               (make-instance 'layer
                                              :neurons neurons 
                                              :size s)))))
        (setf (network-layers net)
              (append ;; (list (make-instance 'layer
               ;;                      :neurons in-ns
               ;;                      :size in-n))
               hidden-layers
               (list (make-instance 'layer
                                    :neurons out-ns
                                    :size out-n)))))
      (loop 
         for n in (l-neurons (car (network-layers net)))
         do (setf (n-weights n) (%weights in-n)))
      (loop
         for (l1 l2 . rest) on (network-layers net) by #'cdr
         while l2
         for s = (l-size l1) then (l-size l1)
         do (loop 
               for n in (l-neurons l2) 
               do (setf (n-weights n) (%weights s))))
      net)))

