pipeline
{
  agent any
  stages{
    stage("Build"){
      steps { 
        echo 'Building the application'
      }
    }
     stage("Testing"){
      steps { 
        echo 'Testing the application'
      }
    }
     stage("INT"){
      steps { 
        echo 'Deploying to INT'
      }
    }
     stage("Staging"){
      steps { 
        echo 'Deploying to Staging'
      }
    }
     stage("Pre-prod"){
      steps { 
        echo 'Deploying to Pre-prod'
      }
    }
     stage("Prod-Mirror"){
      steps { 
        echo 'Deploying to Prod-Mirror'
      }
    }
     stage("Production"){
      steps { 
        echo 'Deploying to Production'
      }
    }
  }
}
