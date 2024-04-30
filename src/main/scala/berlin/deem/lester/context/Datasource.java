package berlin.deem.lester.context;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

// You might also need to define the Datasource annotation in Java
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.METHOD)
public @interface Datasource {
    String name();
    String[] trackProvenanceBy() default {};
}
