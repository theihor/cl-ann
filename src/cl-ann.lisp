(defpackage :cl-ann
  (:use :common-lisp :anaphora :cl-ann/random))

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

(defgeneric activate (n input))

(defmethod activate ((n neuron) input)
  (with-accessors ((b n-bias) (f n-func) (ws n-weights)) n
    (funcall f (+ b (loop
                       for x across input
                       for w across ws
                       sum (* w b))))))

(defclass network ()
  ((input-n :initarg :input-n
            :initform 0
            :accessor input-n)
   (output-n :initarg :output-n
             :initform 0
             :accessor output-n)
   (forward-links :initarg :forward-links
                  :initform (make-hash-table :test #'eq)
                  :accessor forward-links)
   (backward-links :initarg :backward-links
                   :initform (make-hash-table :test #'eq)
                   :accessor backward-links)))

(defun network->dot (net s)
  (with-slots (forward-links) net
    (format s "digraph g {~%")
    (let ((i 0)
          (n->i (make-hash-table :test #'eq)))
      (labels ((%i (n)
                 (aif (gethash n n->i)
                      it
                      (setf (gethash n n->i) (incf i)))))
        (maphash
         (lambda (n ns)
           (loop for n1 in ns do
                (format s "    ~A -> ~A;~%" (%i n) (%i n1))))
         forward-links)))
    (format s "}~%")))

;; (defgeneric propagate-forward (net input))

;; (defmethod propagate-forward ((net network) input)
;;   (let ((layer (net-input-layer net))
;;         (n->input (make-hash-table :test #'eq)))
;;     (loop for n in layer
;;        for x across input
;;        do (push x (gethash n n->input)))
;;     (loop while layer do
;;          (loop
;;             for n in layer
;;             do (let* ((input (map 'simple-vector #'identity
;;                                   (reverse (gethash n n->input))))
;;                       (output (activate n input)))
;;                  (loop for next-n in (next-neurons n) do
;;                       (progn
;;                         (remhash n n->input)
;;                         (push output (gethash next-n n->input))))))
;;          (setf layer (alexandria:hash-table-keys n->input)))))

(defun sigmoid (x)
  (/ 1 (+ 1 (exp (- x)))))

(defun make-random-net (in-n out-n other-n)
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
                            :transfer-function #'sigmoid)))
    (let ((in-ns (loop repeat in-n collect (%neuron)))
          (out-ns (loop repeat out-n collect (%neuron)))
          (other-ns (loop repeat other-n collect (%neuron)))
          (net (make-instance 'network
                              :input-n in-n
                              :output-n out-n)))
      (with-slots (forward-links backward-links) net

        (loop for n in in-ns do
             (let* ((n-out-n (1+ (random other-n)))
                    (n-outs (random-take n-out-n other-ns)))
               (setf (gethash n forward-links) n-outs)
               (loop for n-out in n-outs do
                    (pushnew n (gethash n-out backward-links) :test #'eq))
               
               (setf (n-weights n) (%weights 1))))
        
        (loop for n in other-ns do
             (let* ((n-out-n (1+ (random (+ out-n other-n))))
                    (n-outs (random-take n-out-n (remove n (append out-ns other-ns) :test #'eq))))
               (setf (gethash n forward-links) n-outs)
               (loop for n-out in n-outs do
                    (pushnew n (gethash n-out backward-links) :test #'eq))))
        (loop for n in (append out-ns other-ns) do
             (setf (n-weights n) (%weights (length (gethash n backward-links))))))
      net)))

