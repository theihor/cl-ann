(defpackage :cl-ann/transfer-functions
  (:use :common-lisp)
  (:export #:log-sigmoid
           #:htan-sigmoid
           #:derivative))

(in-package :cl-ann/transfer-functions)

(defparameter *derivatives* (make-hash-table :test #'eq))

(defun log-sigmoid (x)
  (/ 1 (+ 1 (exp (- x)))))

(defun d-log-sigmoid (x)
  (let ((z (log-sigmoid x)))
    (* z (- 1 z))))

(setf (gethash #'log-sigmoid *derivatives*) #'d-log-sigmoid)

(defun htan-sigmoid (x)
  (let ((e+x (exp x))
        (e-x (exp (- x))))
    (/ (- e+x e-x)
       (+ e+x e-x))))

(defun d-htan-sigmoid (x)
  (let ((z (tanh x)))
    (- 1 (* z z))))

(setf (gethash #'htan-sigmoid *derivatives*) #'d-htan-sigmoid)

(defun derivative (f)
  (or (gethash f *derivatives*) 
      (error "No derivative for function ~A" f)))

