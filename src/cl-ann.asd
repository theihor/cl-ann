(asdf:defsystem :cl-ann
  :class :package-inferred-system
  :defsystem-depends-on (:asdf-package-system)
  :pathname #p"./"
  :depends-on (:cl-ann/random)
  :in-order-to ((test-op ;; (load-op :src/test/field)
                         ))
  :perform (test-op (o c)
                    ;; (lisp-unit:run-tests :all :src/test/field)
                    )
  :components ((:file "cl-ann")))

(register-system-packages :spatial-trees '(:rectangles))

