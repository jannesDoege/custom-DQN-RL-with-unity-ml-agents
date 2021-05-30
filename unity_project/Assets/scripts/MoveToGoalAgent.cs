using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class MoveToGoalAgent : Agent
{
    [SerializeField] private Transform targetTrans;
    [SerializeField] private Material winMaterial;
    [SerializeField] private Material loseMaterial;
    [SerializeField] private MeshRenderer floorMeshRenderer;
    private float moveSpeed = 5f;

    public override void OnEpisodeBegin()
    {
        transform.localPosition = new Vector3(Random.Range(4.5f, 1f), 0, Random.Range(-3.5f, 4f));
        targetTrans.localPosition = new Vector3(Random.Range(-4f, -1f), -1f, Random.Range(-3.5f, 4f));
    }
    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(transform.localPosition);
        sensor.AddObservation(targetTrans.localPosition);
    }


    public override void OnActionReceived(ActionBuffers actions)
    {
 
        int forward = actions.DiscreteActions[0];
        int backward = actions.DiscreteActions[1];
        int left = actions.DiscreteActions[2];
        int right = actions.DiscreteActions[3];

        int moveX = forward - backward;
        int moveZ = left - right;

        transform.localPosition += new Vector3(moveX, 0, moveZ) * Time.deltaTime * moveSpeed;
    }

    private void OnTriggerEnter(Collider other){
        if (other.TryGetComponent<Goal>(out Goal goal)){
            SetReward(+1f);
            floorMeshRenderer.material = winMaterial;
            EndEpisode();    
        }
        if (other.TryGetComponent<Wall>(out Wall wall)){
            SetReward(-1f);
            floorMeshRenderer.material = loseMaterial;
            EndEpisode();    
        }         
    }   
}
