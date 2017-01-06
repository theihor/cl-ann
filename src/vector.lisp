(defpackage :cl-ann/vector
  (:use :common-lisp :anaphora :alexandria)
  (:export #:vector-mult
           #:vector-add-to
           #:vector-sub-from
           #:element-wise))

(in-package :cl-ann/vector)

(defun vector-mult (v1 v2)
  (loop
     for x1 across v1
     for x2 across v2
     sum (* x1 x2)))

(defmacro vector-apply-and-store (store-v what-v f)
  (alexandria:with-gensyms (a1 a2)
    `(let ((,a1 ,what-v)
           (,a2 ,store-v))
       (loop 
          for x1 across ,a1
          for x2 across ,a2
          for i from 0
          sum (setf (aref ,a2 i)
                    (funcall ,f x2 x1)))
       ,a2)))

(defun vector-add-to (to-v what-v)
  "Adds what-v to to-v, result is stored in to-v"
  (assert (= (length to-v) (length what-v)))
  (vector-apply-and-store to-v what-v #'+))

(defun vector-sub-from (from-v what-v)
  "Substracts what-v from from-v, result is stored in from-v"
  (assert (= (length from-v) (length what-v)))
  (vector-apply-and-store from-v what-v #'-))

(defmacro element-wise (f v &rest vs)
  `(map 'simple-vector ,f ,@(cons v vs)))

