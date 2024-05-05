package berlin.deem.lester.context

import scala.jdk.CollectionConverters._
import org.reflections.Reflections
import org.reflections.scanners.MethodAnnotationsScanner
import org.reflections.util.{ClasspathHelper, ConfigurationBuilder}
import java.lang.reflect.Method

object AnnotationScanner {

  def scanFor(packageName: String, annotationClass: Class[_ <: java.lang.annotation.Annotation]): (String, String) = {
    val reflections = new Reflections(new ConfigurationBuilder()
      .setUrls(ClasspathHelper.forPackage(packageName))
      .setScanners(new MethodAnnotationsScanner())
    )

    val annotatedMethod = reflections.getMethodsAnnotatedWith(annotationClass)
      .asScala.toArray
      .head

    println(s"  Found @${annotationClass.getName} at ${annotatedMethod.getDeclaringClass.getName}#${annotatedMethod.getName}")
    (annotatedMethod.getDeclaringClass.getName, annotatedMethod.getName)
  }

}
